import os
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
from datetime import datetime, timedelta

def load_signal(file_path,column_name):
    df=pd.read_csv(file_path,sep="\t",header=None)
    if df.shape[1]==1:
        df.columns=[column_name]
        df["time"]=df.index/32
    else:
        df.columns = ["time", column_name]
    return df

def plot_signals(participant_path,participant_name):
    flow=load_signal(os.path.join(participant_path,"flow.txt"),"flow")
    thorac=load_signal(os.path.join(participant_path,"Thorac.txt"),"Thorac")
    spo2=load_signal(os.path.join(participant_path,"SPO2.txt"),"spo2")

    
    flow = flow.iloc[::10].reset_index(drop=True)
    thorac = thorac.iloc[::10].reset_index(drop=True)
    spo2 = spo2.iloc[::2].reset_index(drop=True)
    
    flow = flow[flow["time"] < 600].reset_index(drop=True)
    thorac = thorac[thorac["time"] < 600].reset_index(drop=True)
    spo2 = spo2[spo2["time"] < 600].reset_index(drop=True)
    
    def parse_flow_events(path):
        events_list = []
        start_header = None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Look for header "Start Time: ..."
                if line.lower().startswith('start time:'):
                    hdr = line.split(':', 1)[1].strip()
                    # try a few common formats
                    for fmt in ('%m/%d/%Y %I:%M:%S %p', '%d.%m.%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                        try:
                            start_header = datetime.strptime(hdr, fmt)
                            break
                        except Exception:
                            continue
                    continue
                # lines with events contain a '-' and a ';'
                if '-' in line and ';' in line:
                    try:
                        time_range = line.split(';', 1)[0].strip()
                        start_str, end_str = time_range.split('-', 1)
                        
                        start_dt = None
                        end_dt = None
                        # try parsing start
                        for fmt in ('%d.%m.%Y %H:%M:%S,%f', '%d.%m.%Y %H:%M:%S'):
                            try:
                                start_dt = datetime.strptime(start_str.strip(), fmt)
                                break
                            except Exception:
                                continue
                        if start_dt is None:
                            # fallback: try with day.month.year and space variations
                            try:
                                start_dt = datetime.strptime(start_str.strip(), '%d.%m.%Y %H:%M:%S')
                            except Exception:
                                # give up this line
                                continue

                        end_str = end_str.strip()
                        # if end has a date part (contains '.'), parse directly
                        if '.' in end_str.split()[0]:
                            for fmt in ('%d.%m.%Y %H:%M:%S,%f', '%d.%m.%Y %H:%M:%S'):
                                try:
                                    end_dt = datetime.strptime(end_str, fmt)
                                    break
                                except Exception:
                                    continue
                        else:
                            # append date from start
                            date_part = start_dt.strftime('%d.%m.%Y')
                            combined = f"{date_part} {end_str}"
                            for fmt in ('%d.%m.%Y %H:%M:%S,%f', '%d.%m.%Y %H:%M:%S'):
                                try:
                                    end_dt = datetime.strptime(combined, fmt)
                                    break
                                except Exception:
                                    continue

                        if end_dt is None:
                            continue
                        # handle crossing midnight
                        if end_dt < start_dt:
                            end_dt += timedelta(days=1)

                        events_list.append((start_dt, end_dt))
                    except Exception:
                        continue

        if not events_list:
            return pd.DataFrame(columns=['start', 'end'])

        # baseline: prefer start_header if available, else use the first event start
        baseline = start_header or events_list[0][0]
        rows = []
        for s, e in events_list:
            rows.append({'start': (s - baseline).total_seconds(), 'end': (e - baseline).total_seconds()})
        return pd.DataFrame(rows)

    events = parse_flow_events(os.path.join(participant_path, "flowEvents.txt"))

    fig, axes=plt.subplots(3,1,figsize=(15,10),sharex=True)

    
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        # smaller fonts for performance and compactness
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)

    axes[0].plot(flow["time"],flow["flow"])
    axes[1].plot(thorac["time"],thorac["Thorac"])
    axes[2].plot(spo2["time"],spo2["spo2"],color="green")

    for _, event in events.iterrows():
        # only draw if start/end look numeric
        try:
            s = float(event["start"])
            e = float(event["end"])
        except Exception:
            continue
        for ax in axes:
            ax.axvspan(s, e, color="red", alpha=0.3)

    axes[0].set_title("Nasal Airflow")
    axes[1].set_title("Thoracic Movement")
    axes[2].set_title("Spo2")

    plt.xlabel("Time (seconds)")
    plt.tight_layout()

    os.makedirs("Visualizations",exist_ok=True)
    plt.savefig(f"Visualizations/{participant_name}_visualization.pdf")
    print("Visualization saved")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-name", required=True)
    args=parser.parse_args()

    
    provided = args.name
    if os.path.isabs(provided) and os.path.isdir(provided):
        participant_path = provided
    elif os.path.isdir(provided):
        
        participant_path = provided
    else:
        script_root = os.path.dirname(os.path.dirname(__file__))
        alt = os.path.join(script_root, 'Data', provided)
        if os.path.isdir(alt):
            participant_path = alt
        else:
            
            data_dir = os.path.join(script_root, 'Data')
            available = []
            if os.path.isdir(data_dir):
                available = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            raise SystemExit(
                f"Participant folder not found: '{provided}'.\n"
                f"Tried: '{provided}' and '{alt}'.\n"
                f"Available participants under '{data_dir}': {available}\n"
                "Pass either the full path or a participant id (e.g. AP01)."
            )

    participant_name = os.path.basename(participant_path.rstrip(os.sep))
    plot_signals(participant_path, participant_name)

