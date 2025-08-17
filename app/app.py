import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter.font import Font
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import time
import threading
import cv2

# Import your custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glcm.resnet_glcm import ResNetWithInternalGLCM
from inference.explainer import explain_prediction
from inference.checker import is_retinal_image_openai

"""

V1 of the the app which was made using tkinter for proof of concept !!
"""
class RetinalAnomalyDetector:
    def __init__(self, model_path, labels_csv, device=None):
        """Initialize the retinal anomaly detector with model and labels."""
        self.device = device or (torch.device('cuda:0')) if torch.cuda.is_available() else torch.device('cpu')
        self.labels = self._load_labels(labels_csv)
        self.model = self._load_model(model_path, len(self.labels), self.device)
        self.transform = self._get_transforms()
        self.load_time = time.time()
        
        # Grad-CAM attributes
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def apply_normal_constraint(self, probs, normal_idx):
        """
        Enforce the constraint that if 'normal' is predicted positive, all diseases are set to zero.
        probs: numpy array of shape (num_classes,), after sigmoid
        normal_idx: index of the 'normal' class
        """
        probs = probs.copy()  # avoid in-place modification
        
        if probs[normal_idx] > 0.8:
            probs[:normal_idx] = 0.0
        return probs

    def _register_hooks(self):
        """Register hooks to capture gradients and activations for Grad-CAM."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # Register hooks on the last convolutional layer
        target_layer = self.model.base_model.layer4[-1].conv3
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def _load_labels(self, csv_path):
        """Load class labels from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            return df.columns[1:].tolist()
        except Exception as e:
            raise ValueError(f"Failed to load labels from {csv_path}: {str(e)}")

    def _load_model(self, model_path, num_classes, device):
        """Load and initialize the model."""
        try:
            model = ResNetWithInternalGLCM(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle potential state dict mismatches
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

    def _get_transforms(self):
        """Get image transformations for model input."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """Predict anomalies from retinal image."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # First check if it's a retinal scan
        if not is_retinal_image_openai(image_path): 
            return None, "Not a retinal scan"

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.sigmoid(outputs).cpu().squeeze(0).numpy()
                probs = self.apply_normal_constraint(probs, normal_idx=len(self.labels) - 1)

            results = list(zip(self.labels, probs))
            results.sort(key=lambda x: x[1], reverse=True)
            return results, None
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def generate_gradcam(self, input_tensor, original_image=None, class_idx=None):
        """Generate Grad-CAM heatmap for the input image."""
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        
        # Create one-hot encoded tensor for backprop
        one_hot = torch.zeros_like(outputs)
        one_hot[0][class_idx] = 1
        outputs.backward(gradient=one_hot)
        
        # Pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations
        activations = self.activations[0].cpu().detach()
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0)
        heatmap = torch.relu(heatmap)  # Only keep positive influences
        heatmap /= torch.max(heatmap)  # Normalize
        
        # Convert to numpy and apply colormap
        heatmap = heatmap.numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image for blending
        img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.uint8(255 * img)
        
        # Resize both images to match original dimensions if provided
        if original_image is not None:
            original_size = original_image.size
            heatmap = cv2.resize(heatmap, original_size)
            img = cv2.resize(img, original_size)
        
        # Blend heatmap with original image
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        return superimposed_img

    @staticmethod
    def explain_prediction(results, threshold=0.5):
        """Generate human-readable explanation of prediction results."""
        if not results:
            return "No prediction results available."

        top_label, top_prob = results[0]
        explanation = [
            f"Primary finding: {top_label} (confidence: {top_prob:.1%})",
            "",
            "Additional findings:"
        ]
        
        # Add all findings above threshold
        significant_findings = [(label, prob) for label, prob in results if prob >= threshold]
        
        if len(significant_findings) > 1:
            for label, prob in significant_findings[1:]:
                explanation.append(f"- {label} (confidence: {prob:.1%})")
        else:
            explanation.append("- No other significant findings detected")
            
        if top_prob < threshold:
            explanation.append("\nNote: Confidence is below diagnostic threshold. This may represent:")
            explanation.append("- Early or mild manifestation")
            explanation.append("- Image quality limitations")
            explanation.append("- Normal anatomical variation")
            
        return "\n".join(explanation)


class InferenceApp(tk.Tk):
    def __init__(self, detector):
        """Initialize the GUI application."""
        super().__init__()
        self.detector = detector
        self.title("Retinal Anomaly Detector")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.configure(bg="#f0f0f0")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('TLabel', font=('Helvetica', 9), background="#f0f0f0")
        self.style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))
        
        # Font for results
        self.result_font = Font(family="Courier", size=10)
        
        # Image display variables
        self.current_image = None
        self.gradcam_image = None
        self.original_image = None
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image selection section
        selection_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        selection_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(selection_frame)
        btn_frame.pack(fill=tk.X)
        
        self.select_btn = ttk.Button(btn_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(btn_frame, text="Run Analysis", 
                                    command=self.run_prediction, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Image path display
        self.image_path_var = tk.StringVar()
        path_frame = ttk.Frame(selection_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="Selected:").pack(side=tk.LEFT)
        self.image_path_label = ttk.Label(path_frame, textvariable=self.image_path_var, 
                                         wraplength=650)
        self.image_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Image analysis section
        analysis_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding="10")
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a container frame for the images
        img_container = ttk.Frame(analysis_frame)
        img_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        orig_frame = ttk.LabelFrame(img_container, text="Original Image", padding="5")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.preview_label = ttk.Label(orig_frame, background="white")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Grad-CAM frame
        gradcam_frame = ttk.LabelFrame(img_container, text="Attention Heatmap (Grad-CAM)", padding="5")
        gradcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.gradcam_label = ttk.Label(gradcam_frame, background="white")
        self.gradcam_label.pack(fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_box = scrolledtext.ScrolledText(
            results_frame, 
            width=80, 
            height=15, 
            font=self.result_font,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.results_box.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5,0))

    def select_image(self):
        """Handle image selection."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
            ("JPEG Images", "*.jpg *.jpeg"), 
            ("PNG Images", "*.png"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select retinal image", 
            filetypes=filetypes
        )
        
        if filepath:
            self.image_path_var.set(filepath)
            self.predict_btn.config(state=tk.NORMAL)
            self.clear_results()
            self.display_image_preview(filepath)

    def display_image_preview(self, image_path):
        """Display a thumbnail preview of the selected image."""
        try:
            self.original_image = Image.open(image_path)
            img = self.original_image.copy()
            
            # Calculate reduced size for display
            original_width, original_height = img.size
            new_width = int(original_width * 0.6)
            new_height = int(original_height * 0.6)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.current_image)
            
            # Initialize blank Grad-CAM placeholder at same reduced size
            blank = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            self.gradcam_placeholder = ImageTk.PhotoImage(blank)
            self.gradcam_label.config(image=self.gradcam_placeholder)
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not display image: {str(e)}")

    def clear_results(self):
        """Clear the results display."""
        self.results_box.config(state=tk.NORMAL)
        self.results_box.delete('1.0', tk.END)
        self.results_box.config(state=tk.DISABLED)

    def run_prediction(self):
        """Run prediction on selected image."""
        image_path = self.image_path_var.get()
        if not image_path:
            messagebox.showwarning("No image selected", "Please select an image first.")
            return

        # Disable buttons during prediction
        self.select_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing image...")
        self.update_idletasks()
        
        # Run prediction in a separate thread to keep UI responsive
        threading.Thread(
            target=self._run_prediction_thread, 
            args=(image_path,),
            daemon=True
        ).start()

    def _run_prediction_thread(self, image_path):
        """Thread worker for prediction task."""
        try:
            start_time = time.time()
            
            # First check if it's a retinal scan
            results, error = self.detector.predict(image_path)
            
            if error == "Not a retinal scan":
                self.after(0, self._display_non_retinal_result, error, start_time)
                return
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.detector.transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(self.detector.device)
            
            # Generate Grad-CAM with original image size
            gradcam_img = self.detector.generate_gradcam(input_tensor, self.original_image)
            gradcam_img = Image.fromarray(gradcam_img)
            
            # Reduce both images by 30%
            original_width, original_height = self.original_image.size
            new_width = int(original_width * 0.6)
            new_height = int(original_height * 0.6)
            gradcam_img = gradcam_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Prepare results
            explanation = self.detector.explain_prediction(results)
            elapsed = time.time() - start_time
            
            # Update UI in main thread
            self.after(0, self._display_results, 
                      results, explanation, elapsed, gradcam_img)
            
        except Exception as e:
            self.after(0, self._handle_prediction_error, str(e))
            
        finally:
            self.after(0, self._enable_buttons)

    def _display_non_retinal_result(self, error, start_time):
        """Display message for non-retinal images."""
        elapsed = time.time() - start_time
        
        self.results_box.config(state=tk.NORMAL)
        self.results_box.delete('1.0', tk.END)
        self.results_box.insert(tk.END, "IMAGE ANALYSIS REPORT\n", "header")
        self.results_box.insert(tk.END, "="*50 + "\n\n")
        self.results_box.insert(tk.END, f"ERROR: {error}\n\n", "error")
        self.results_box.insert(tk.END, "The selected image does not appear to be a retinal scan.\n")
        self.results_box.insert(tk.END, "Please select a valid retinal fundus image for analysis.\n")
        self.results_box.insert(tk.END, f"\nProcessing time: {elapsed:.2f} seconds\n")
        
        self.results_box.tag_config("header", font=("Helvetica", 11, "bold"))
        self.results_box.tag_config("error", foreground="red")
        self.results_box.config(state=tk.DISABLED)
        
        # Clear Grad-CAM display
        blank = Image.new('RGB', (300, 300), (255, 255, 255))
        self.gradcam_placeholder = ImageTk.PhotoImage(blank)
        self.gradcam_label.config(image=self.gradcam_placeholder)
        
        self.status_var.set(f"Analysis completed in {elapsed:.2f} seconds")

    def _display_results(self, predictions, explanation, elapsed_time, gradcam_img):
        """Display prediction results in the UI."""
        # Display Grad-CAM
        self.gradcam_display = ImageTk.PhotoImage(gradcam_img)
        self.gradcam_label.config(image=self.gradcam_display)
        
        # Update results text
        self.results_box.config(state=tk.NORMAL)
        self.results_box.delete('1.0', tk.END)
        
        # Add header
        self.results_box.insert(tk.END, "RETINAL ANALYSIS REPORT\n", "header")
        self.results_box.insert(tk.END, "="*50 + "\n\n")
        
        # AI prediction summary
        self.results_box.insert(tk.END, "FINDINGS SUMMARY:\n", "bold")
        summary = str(explain_prediction(predictions))
        self.results_box.insert(tk.END, summary + "\n\n")

        # Add predictions
        self.results_box.insert(tk.END, "DETECTED FINDINGS (confidence score):\n", "bold")
        for label, prob in predictions:
            self.results_box.insert(tk.END, f"- {label:<25}: {prob:.1%}\n")
        
        # Add explanation
        self.results_box.insert(tk.END, "\nCLINICAL INTERPRETATION:\n", "bold")
        self.results_box.insert(tk.END, explanation + "\n")
        
        # Add processing time
        self.results_box.insert(tk.END, f"\nProcessing time: {elapsed_time:.2f} seconds\n")
        
        # Configure tags for formatting
        self.results_box.tag_config("header", font=("Helvetica", 11, "bold"))
        self.results_box.tag_config("bold", font=("Helvetica", 10, "bold"))
        
        self.results_box.config(state=tk.DISABLED)
        self.status_var.set(f"Analysis completed in {elapsed_time:.2f} seconds")

    def _handle_prediction_error(self, error_msg):
        """Handle errors during prediction."""
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("Error during processing")

    def _enable_buttons(self):
        """Re-enable UI buttons after prediction completes."""
        self.select_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)

    def on_close(self):
        """Handle application close."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.destroy()


def main():
    #Main application entry point.
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