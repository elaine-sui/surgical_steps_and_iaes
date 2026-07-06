
import torch
from tqdm import tqdm
import numpy as np
from torchmetrics import AveragePrecision as AP
from torchmetrics.classification import BinaryAUROC
import os
import wandb
import math
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from timm.utils import accuracy as timm_accuracy
import pickle

from .metrics import ANETdetection, keyframe_distance

def move_to_cuda(data, bf16=False):
    stack = [(data, [])]
    while stack:
        current_data, path = stack.pop()
        if isinstance(current_data, dict) or isinstance(current_data,BatchFeature):
            for key, value in current_data.items():
                stack.append((value, path + [key]))
        elif isinstance(current_data, list):
            for idx, value in enumerate(current_data):
                stack.append((value, path + [idx]))
        elif isinstance(current_data, torch.Tensor):
            tensor = current_data.cuda()
            if bf16:
                tensor = tensor.to(torch.bfloat16)
            temp_data = data
            for p in path[:-1]:
                temp_data = temp_data[p]
            temp_data[path[-1]] = tensor

    return data

def train_and_evaluate(model,train_loader,val_loader,optimizer,save_path,num_epochs=5,scheduler=None,use_bf16=False,test_loader=None,accumulation_steps=1, save_pred_and_gt=False, monitor='auroc', fps=1):
    wandb.init(project="Error_Detection", name='/'.join(save_path.split("/")[-2:]))
    best_metric = 0.0
    best_ckpt = None
    os.makedirs(save_path,exist_ok=True)

    print_interval = 100 # len(train_loader) // 5
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_loc = 0.0
        for i, (images, labels, ids) in enumerate(train_loader):   
            if isinstance(labels, tuple):
                labels=[label.to("cuda").round().long() for label in labels] 
            else:       
                labels=labels.to("cuda").round().long()
            outputs = model(move_to_cuda(images,bf16=use_bf16))
            loss = model.loss(outputs, labels)

            loss_cls, loss_loc = None, None
            if isinstance(loss, tuple):
                loss, loss_cls, loss_loc = loss

            loss = loss / accumulation_steps
            if math.isnan(loss):
                continue
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step() 
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if loss_cls is not None:
                running_loss_cls += loss_cls.item()
                running_loss_loc += loss_loc.item()
            running_loss += loss.item() * accumulation_steps

            if (i + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {(running_loss/(i+1)):.4f}')
            
        print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {running_loss / len(train_loader):.4f}')

        val_metrics_dict = evaluate(model,val_loader,use_bf16=use_bf16, save_pred_and_gt=save_pred_and_gt,metric=monitor, fps=fps)
        ckpt_path = save_path+f"/model_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        test_metrics_dict = {}
        if test_loader is not None:
            test_metrics_dict = evaluate(model,test_loader,save_path=ckpt_path.replace("/",""),use_bf16=use_bf16,metric=monitor, fps=fps)
        
        wandb_dict = {"loss": running_loss / len(train_loader), "epoch": epoch}

        if loss_cls is not None:
            wandb_dict.update({'loss_cls': running_loss_cls / len(train_loader), 'loss_loc': running_loss_loc/len(train_loader)})
        wandb_dict.update({f'val/{key}': val for key,val in val_metrics_dict.items()})
        wandb_dict.update({f'test/{key}': val for key,val in test_metrics_dict.items()})
        wandb.log(wandb_dict)

        val_metric = val_metrics_dict[monitor]
        if val_metric > best_metric:
            best_metric = val_metric
            best_ckpt = ckpt_path
            
            print(f'Best model saved with {monitor}: {val_metric:.4f}')
    
    print("="*80)
    print(f"Best checkpoint: {best_ckpt}")
    print("="*80)
    return best_ckpt

def evaluate(model,test_loader,save_path=None,use_bf16=False,save_pred_and_gt=False, sliding_window=False, metric=None, fps=1):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits=[]
    all_probs=[]
    all_loc_logits=[]

    running_loss_cls = 0.0
    running_loss_loc = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels, ids in tqdm(test_loader):
            predicted,probs,logits = model.predict(move_to_cuda(images,bf16=use_bf16))

            if isinstance(labels, tuple):
                labels=[label.to("cuda").round().long() for label in labels] 
            else:       
                labels=labels.to("cuda").round().long()

            loss = model.loss(logits, labels)

            loss_cls, loss_loc = None, None
            if isinstance(loss, tuple):
                loss, loss_cls, loss_loc = loss

            running_loss += loss.item()

            if loss_cls is not None:
                running_loss_cls += loss_cls.item()
                running_loss_loc += loss_loc.item()

            if isinstance(labels, tuple) or isinstance(labels, list):
                labels=[label.cpu().numpy() for label in labels]
                predicted=[predict.cpu().numpy() for predict in predicted]
                probs=[prob.cpu().numpy() for prob in probs]
            else:       
                labels=labels.cpu().numpy()
                predicted=predicted.cpu().numpy()
                probs=probs.cpu().numpy()
                
            all_labels.append(labels)
            all_preds.append(predicted)
            all_probs.append(probs)

            if isinstance(logits, tuple):
                for id,cls_logit,loc_logit,cls_prob,loc_prob,cls_label,loc_label in zip(ids,logits[0],logits[1],all_probs[-1][0],all_probs[-1][1],all_labels[-1][0],all_labels[-1][1]):
                    all_logits.append({'id': id, 'cls_logits': cls_logit.squeeze().to(float), 'cls_pred': cls_prob, 'cls_label': cls_label, 'loc_logits': loc_logit.squeeze().to(float), 'loc_pred': loc_prob, 'loc_label': loc_label})
                    all_loc_logits.append(loc_logit.unsqueeze(0).to(float))
            else:
                for id,logit,prob,label in zip(ids,logits,all_probs[-1],all_labels[-1]):
                    all_logits.append({'id': id, 'logits': logit.squeeze().to(float), 'pred': prob, 'label': label})

    if save_pred_and_gt:
        if sliding_window:
            pred_and_gt_save_path = "./pred_and_gt_sliding_window/"+save_path
        else:
            pred_and_gt_save_path = "./pred_and_gt/"+save_path
        parent = os.path.split(pred_and_gt_save_path)[0]
        os.makedirs(parent, exist_ok=True)
        with open(pred_and_gt_save_path,"wb") as file:
            pickle.dump(all_logits, file)
        
        print(f"Saved pred and gt to {pred_and_gt_save_path}")

    if 'cls_logits' in all_logits[0]:
        all_cls_labels = [label[0] for label in all_labels]
        all_loc_labels = [label[1] for label in all_labels]
        all_cls_preds = [pred[0] for pred in all_preds]
        all_loc_preds = [pred[1] for pred in all_preds]
        all_cls_probs = [prob[0] for prob in all_probs]
        all_loc_probs = [prob[1] for prob in all_probs]

        all_cls_labels = torch.tensor(np.concatenate(all_cls_labels)).round().long()
        all_cls_preds =torch.tensor(np.concatenate(all_cls_preds))
        all_cls_probs = torch.tensor(np.concatenate(all_cls_probs))

        all_loc_labels = torch.tensor(np.concatenate(all_loc_labels)).round().long()
        all_loc_preds =torch.tensor(np.concatenate(all_loc_preds))
        all_loc_probs = torch.tensor(np.concatenate(all_loc_probs))

        # Classification metrics
        cls_acc = (all_cls_labels == all_cls_preds).sum()/all_cls_labels.shape[0]
        cls_auroc = BinaryAUROC()(all_cls_probs, all_cls_labels)
        print(f'Classification Accuracy: {cls_acc} \t Classification AUROC: {cls_auroc}')
        metrics_dict = {'accuracy': cls_acc, 'auroc': cls_auroc}

        # Localization metrics
        all_loc_logits = torch.concatenate(all_loc_logits).cpu()

        # Only compute localization on the examples with state change
        all_loc_logits = all_loc_logits[all_cls_labels == 1]
        all_loc_labels = all_loc_labels[all_cls_labels == 1]
        loc_acc1, loc_acc5 = timm_accuracy(all_loc_logits, all_loc_labels, topk=(1, 5))
        print(f'Localization Acc @1: {loc_acc1} \t Localization Acc @5: {loc_acc5}')
        metrics_dict.update({'loc_accuracy@1': loc_acc1.item(), 'loc_accuracy@5': loc_acc5.item()})

        # PNR distance
        avg_distance = keyframe_distance(all_loc_probs, all_loc_labels, all_cls_labels, fps)
        print(f'PNR Distance: {avg_distance}')
        metrics_dict.update({'pnr_distance': avg_distance.item()})
    else:
        all_labels = torch.tensor(np.concatenate(all_labels)).round().long()
        all_preds =torch.tensor(np.concatenate(all_preds))
        all_probs = torch.tensor(np.concatenate(all_probs))

        if len(all_labels.shape) > 1 and all_labels.shape[1] == 4:
            blood_masks = all_labels[:, 2]
            bile_masks = all_labels[:, 3]
            blood_labels = all_labels[:, 0][blood_masks == 1]
            bile_labels = all_labels[:, 1][bile_masks == 1]
            blood_preds = all_preds[:, 0][blood_masks == 1]
            bile_preds = all_preds[:, 1][bile_masks == 1]
            blood_probs = all_probs[:, 0][blood_masks == 1]
            bile_probs = all_probs[:, 1][bile_masks == 1]

            blood_acc = (blood_labels == blood_preds).sum() / blood_labels.shape[0]
            blood_auroc = BinaryAUROC()(blood_probs, blood_labels)
            bile_acc = (bile_labels == bile_preds).sum() / bile_labels.shape[0]
            bile_auroc = BinaryAUROC()(bile_probs, bile_labels)

            print(f'Blood Accuracy: {blood_acc} \t Bile Accuracy: {bile_acc}')
            print(f'Blood AUROC: {blood_auroc} \t Bile AUROC: {bile_auroc}')

            metrics_dict = {'accuracy_blood': blood_acc, 'accuracy_bile': bile_acc, 'auroc_blood': blood_auroc, 'auroc_bile': bile_auroc, 'accuracy': (blood_acc + bile_acc) / 2, 'auroc': (blood_auroc + bile_auroc) / 2}
        else:

            acc = (all_labels == all_preds).sum()/all_labels.shape[0]
            auroc = BinaryAUROC()(all_probs, all_labels)
            print(f'Accuracy: {acc} \t AUROC: {auroc}')

            metrics_dict = {'accuracy': acc, 'auroc': auroc}

            if metric == 'map':
                det = ANETdetection(all_logits)
                ap_metrics_dict, average_mAP = det.evaluate()
                metrics_dict.update(ap_metrics_dict)
                metrics_dict['average_mAP'] = average_mAP
    
    metrics_dict['loss'] = running_loss / len(test_loader)
    metrics_dict['loss_loc'] = running_loss_loc / len(test_loader)
    metrics_dict['loss_cls'] = running_loss_cls / len(test_loader)

    return metrics_dict

def extract_features(model, data_loader, features_path, use_bf16=False):
    if os.path.exists(features_path):
        print(f"{features_path} already saved")
        return
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for images, labels, ids in tqdm(data_loader):
            embeds = model.get_embedding_peft(move_to_cuda(images,bf16=use_bf16)).detach().cpu().numpy()
            all_embeds.append(embeds)
    
    all_embeds = np.concatenate(all_embeds)
    np.save(features_path, all_embeds)
    print(f"Saved to {features_path}")


def predict(model,test_loader,use_bf16=False,error_type=None):
    model.eval()
    all_logits=[]
    with torch.no_grad():
        for images, labels, ids in tqdm(test_loader):
            predicted,probs,logits = model.predict(move_to_cuda(images,bf16=use_bf16))
        
            predicted=predicted.cpu().numpy()
            probs=probs.cpu().numpy()

            if error_type is not None:
                if error_type.startswith('perforation') or error_type.startswith('bile'):
                    probs = probs[:, 1]
                else:
                    probs = probs[:, 0]

            for id,prob in zip(ids,probs):
                all_logits.append({'id': id, 'pred': prob.item()})

    return all_logits

def predict_multiclass(model,test_loader,use_bf16=False):
    model.eval()
    all_blood_logits=[]
    all_bile_logits=[]
    with torch.no_grad():
        for images, labels, ids in tqdm(test_loader):
            predicted,probs,logits = model.predict(move_to_cuda(images,bf16=use_bf16))
        
            predicted=predicted.cpu().numpy()
            probs=probs.cpu().numpy()

            for id,prob in zip(ids,probs):
                all_blood_logits.append({'id': id, 'pred': prob[0].item()})
                all_bile_logits.append({'id': id, 'pred': prob[1].item()})

    return all_blood_logits, all_bile_logits
        
