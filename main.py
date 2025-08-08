from app import RetinalAnomalyDetector, InferenceApp

def main():
    """Main application entry point."""
    try:
        MODEL_PATH = 'models/v4_less_strict.pth'
        LABELS_CSV = 'data/train/train.csv'
        
        print("Initializing retinal anomaly detector...")
        detector = RetinalAnomalyDetector(MODEL_PATH, LABELS_CSV)
        
        print("Starting application...")
        app = InferenceApp(detector)
        app.mainloop()
        
    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")
        raise


if __name__ == "__main__":
    main()