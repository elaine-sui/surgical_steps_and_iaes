import pandas as pd
import argparse

def main(args):
    df = pd.read_csv(args.active_surgery_predictions)
    labels = set(df['label'].values)

    if len(labels) == 1 and 'background' in labels: # all background
        df.loc[0, 'label'] = 'activeSurgery'
        df.to_csv(args.active_surgery_predictions, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--active_surgery_predictions', type=str)

    args = parser.parse_args()
    main(args)