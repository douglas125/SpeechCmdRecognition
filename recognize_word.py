import numpy as np
import tensorflow as tf
import librosa
import SpeechModels
import sys
import os
import argparse

# Configuration
sr = 16000
nCategs = 36

# Category labels
categories = ['nine', 'yes', 'no', 'up', 'down', 'left',
              'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two',
              'three', 'four', 'five', 'six', 'seven', 'eight',
              'backward', 'bed', 'bird', 'cat', 'dog', 'follow',
              'forward', 'happy', 'house', 'learn', 'marvin', 'sheila',
              'tree', 'visual', 'wow']

def load_model():
    """Load and compile the speech recognition model."""
    # Create model
    model = SpeechModels.AttRNNSpeechModel(nCategs)
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load weights
    model.load_weights('model-attRNN.h5')
    
    print("Model loaded successfully!")
    model.summary()
    
    return model

def preprocess_audio(audio_file):
    """Preprocess audio file for prediction."""
    try:
        # Load audio file
        data, _ = librosa.load(audio_file, sr=sr)
        
        # Check if audio is empty
        if len(data) == 0:
            print(f"Warning: Audio file '{audio_file}' is empty")
            return None
        
        # Normalize audio length to 1 second (16000 samples)
        if len(data) > sr:
            # Truncate if longer than 1 second
            data = data[:sr]
        elif len(data) < sr:
            # Pad with zeros if shorter than 1 second
            data = np.pad(data, (0, sr - len(data)), 'constant')
        
        # Add batch dimension
        data = np.expand_dims(data, axis=0)
        
        return data
        
    except Exception as e:
        print(f"Error processing audio file '{audio_file}': {str(e)}")
        return None

def predict_audio(model, audio_data):
    """Make prediction on preprocessed audio data."""
    try:
        # Make prediction
        predictions = model.predict(audio_data)
        
        # Get predicted class and confidence
        predicted_class_index = np.argmax(predictions[0]) - 1
        confidence = np.max(predictions[0])
        
        # Map to category label (fixed the index adjustment bug)
        if 0 <= predicted_class_index < len(categories):
            predicted_label = categories[predicted_class_index]
        else:
            predicted_label = "unknown"
        
        return predicted_label, confidence, predicted_class_index
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None, None

def recognize_files(audio_files):
    """Recognize speech commands in multiple audio files."""
    # Load model
    model = load_model()
    
    results = []
    
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file}")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"Error: File '{audio_file}' not found")
            continue
        
        # Preprocess audio
        audio_data = preprocess_audio(audio_file)
        if audio_data is None:
            continue
        
        # Make prediction
        predicted_label, confidence, class_index = predict_audio(model, audio_data)
        
        if predicted_label is not None:
            result = {
                'file': audio_file,
                'predicted_label': predicted_label,
                'confidence': float(confidence),
                'class_index': class_index
            }
            results.append(result)
            
            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Class Index: {class_index}")
        else:
            print(f"Failed to process {audio_file}")
    
    return results

def main():
    """Main function to handle command line arguments and run recognition."""
    parser = argparse.ArgumentParser(description='Speech Command Recognition')
    parser.add_argument('files', nargs='+', help='Audio files to recognize (one or more)')
    parser.add_argument('--output', '-o', help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Recognize speech commands
    results = recognize_files(args.files)
    
    # Print summary
    print(f"\n{'='*50}")
    print("RECOGNITION SUMMARY")
    print(f"{'='*50}")
    
    for result in results:
        print(f"File: {result['file']}")
        print(f"  Predicted: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print()
    
    # Save results to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write("Speech Command Recognition Results\n")
                f.write("="*40 + "\n\n")
                
                for result in results:
                    f.write(f"File: {result['file']}\n")
                    f.write(f"Predicted: {result['predicted_label']}\n")
                    f.write(f"Confidence: {result['confidence']:.4f}\n")
                    f.write(f"Class Index: {result['class_index']}\n")
                    f.write("-" * 30 + "\n")
            
            print(f"Results saved to: {args.output}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()