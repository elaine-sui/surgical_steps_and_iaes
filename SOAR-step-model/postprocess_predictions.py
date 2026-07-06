import pandas as pd
import argparse
import json

def naive_postprocessing(df):
    """
    Naive postprocessing based on heuristics
    """
    # Cut the df to end at the first step 6 occurrence
    step6 = '6PackagingAndRemoval'
    last_index = len(df)
    if step6 in df['label'].values:
        # Find the last index where the value occurs
        last_index = df[df['label'] == step6].index.max()
    
    # Cut the df to start at the first occurrence of the lowest step number
    df['step_num'] = df.apply(lambda x: x['label'][0], axis=1)
    lowest_step_num = df['step_num'].values.min()
    
    # Find the first index where the step number occurs
    first_index = df[df['step_num'] == lowest_step_num].index.min()

    # Slice the DataFrame up to the last occurrence of the value
    df = df.loc[first_index:last_index]
    df = df.drop(columns=['step_num'])

    return df


def convert_framenum_to_time(frame_num: int, fps: int):
    seconds = frame_num / fps
    hours = int(seconds / 60**2)
    seconds -= hours * 60**2
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    seconds = int(seconds)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def shift_durations(duration_lst, start_seconds):
    new_duration_lst = []
    start_times = []
    end_times = []

    for duration_str in duration_lst:
        start,end = duration_str.split('-')
        start, end = int(start) + start_seconds, int(end) + start_seconds
        
        duration_str = f'{start}-{end}'

        start_time = convert_framenum_to_time(start, 1)
        end_time = convert_framenum_to_time(end, 1)

        new_duration_lst.append(duration_str)
        start_times.append(start_time)
        end_times.append(end_time)
    return new_duration_lst, start_times, end_times


def oob_postprocessing(df, oob_json):
    with open(oob_json, 'r') as f:
        oob_results = json.load(f) # note: this is full video FPS
    
    start, fps = oob_results['start'], oob_results['fps']

    # Shift all the times by start (in seconds)
    start_seconds = int(start / fps)
    durations, start_times, end_times = shift_durations(df['duration'].values, start_seconds)

    df['duration'] = durations
    df['start_time'] = start_times
    df['end_time'] = end_times

    return df


def main(args):
    df = pd.read_csv(args.pred_file)
    output_file = args.pred_file.replace('.csv', '_postprocessed.csv')

    # Note: intervals in the df are exclusive!
    if args.oob_json:
        print("Post process with OOB")
        df = oob_postprocessing(df, args.oob_json)
    
    df = naive_postprocessing(df)
    df.to_csv(output_file, index=False)
    
    print(f'Dumped post-processed predictions to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default=None)
    parser.add_argument('--oob_json', type=str, default=None)
    args = parser.parse_args()

    main(args)