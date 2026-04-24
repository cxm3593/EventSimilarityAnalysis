from event_data_toolbox.event_data_manager import EventDataManager
from event_analysis_toolbox.mmd import mmd_analysis
import yaml  # pyright: ignore[reportMissingModuleSource]

def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():

    config = load_config()
    
    event_data_manager = EventDataManager()

    real_data = event_data_manager.load_event_data_h5(config['real_data_path'], dataset_name="events", data_key="real_data")
    print(f"Real data loaded, total number of events: {real_data.shape[0]}, duration: {real_data['t'].max()} microseconds")
    v2e_data = event_data_manager.load_event_data_h5(config['v2e_data_path'], dataset_name="events", data_key="v2e_data")
    print(f"V2E data loaded, total number of events: {v2e_data.shape[0]}, duration: {v2e_data['t'].max()} microseconds")

    real_0_3ms = event_data_manager.select_events_by_time_window(real_data, 0, 300_000) # Select a window of 30 milliseconds
    print(f"Selected 0-30ms window: {real_0_3ms.shape[0]} events")

    v2e_0_3ms = event_data_manager.select_events_by_time_window(v2e_data, 0, 300_000) # Select a window of 30 milliseconds
    print(f"Selected 0-30ms window: {v2e_0_3ms.shape[0]} events")


    mmd = mmd_analysis(
        real_0_3ms,
        v2e_0_3ms,
        chunk_size=20_000,
        sigma=1.0,
        feature_names=None,
        backend="cupy",
    )
    print(f"MMD analysis: {mmd['mmd']}")

if __name__ == "__main__":
    main()
