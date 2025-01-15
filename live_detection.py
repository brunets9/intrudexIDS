import numpy as np
from scapy.all import sniff, get_if_list, wrpcap, IP, TCP, UDP
import joblib
import datetime
import os
import pandas as pd
import threading
import logging
import yaml

# Load the settings from settings.yaml
def load_config(config_path="configs/settings.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    logging.info(f"Loading model from {model_path} and scaler from {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Extract features dynamically
def extract_features(packet):
    feature_values = {col: 0 for col in joblib.load('models/features_used.pkl')}

    try:
        if packet.haslayer(IP):
            feature_values[' Flow Duration'] = packet[IP].ttl
            if packet.haslayer(TCP):
                feature_values[' Destination Port'] = packet[TCP].dport
                feature_values[' Total Fwd Packets'] = 1
                feature_values[' Fwd Packet Length Max'] = len(packet[TCP].payload)
            elif packet.haslayer(UDP):
                feature_values[' Destination Port'] = packet[UDP].dport
                feature_values[' Total Backward Packets'] = 1
                feature_values['Bwd Packet Length Max'] = len(packet[UDP].payload)
    except Exception as e:
        logging.error(f"Error extracting features from packet: {e}")

    return feature_values

# Process packet and predict
def process_packet(packet, model, scaler, threshold=0.05):
    try:
        if not packet.haslayer(IP):
            return

        logging.info(f"Processing packet from {packet[IP].src} to {packet[IP].dst}")
        feature_values = extract_features(packet)
        feature_names = joblib.load('models/features_used.pkl')
        features_df = pd.DataFrame([feature_values]).reindex(columns=feature_names, fill_value=0)

        features_normalized = scaler.transform(features_df)
        prediction_proba = model.predict_proba(features_normalized)[0]

        intrusion_class_index = np.argmax(prediction_proba[1:]) + 1
        max_probability = prediction_proba[intrusion_class_index]

        if max_probability >= threshold:
            logging.info(f"INTRUSION DETECTED: Class {intrusion_class_index} with probability {max_probability:.2f}, from {packet[IP].src} to {packet[IP].dst}")
            save_packet_to_file(packet)

    except Exception as e:
        logging.error(f"Error processing packet: {e}")

# Save the packet
def save_packet_to_file(packet):
    capture_dir = "data/live_captures"
    os.makedirs(capture_dir, exist_ok=True)
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pcap"
    file_path = os.path.join(capture_dir, filename)
    wrpcap(file_path, packet)
    logging.info(f"Suspected packet saved at {file_path}")

# Capture packets from an interface
def capture_packet_on_interface(interface, model, scaler, packet_count, threshold):
    sniff(
        prn=lambda packet: process_packet(packet, model, scaler, threshold),
        iface=interface,
        count=packet_count,
        promisc=True
    )

# Main function for real-time detection
def live_capture(settings, model_path, scaler_path, threshold):
    interfaces = get_if_list()
    packet_count = settings['real_time_detection']['packet_count']
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    threads = []
    for interface in interfaces:
        thread = threading.Thread(
            target=capture_packet_on_interface,
            args=(interface, model, scaler, packet_count, threshold),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# Entry point
def live_main(config, threshold):
    log_directory = config['paths']['logs']
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, "intrusion_detection.log")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    model_path = config['paths']['model_save_path'] + "intrudex.pkl"
    scaler_path = config['paths']['model_save_path'] + "scaler.pkl"
    logging.info("Starting real-time traffic detection...")
    live_capture(config, model_path, scaler_path, threshold)

if __name__ == "__main__":
    config = load_config()
    threshold = 0.7
    live_main(config, threshold)
