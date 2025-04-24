#!/usr/bin/env python3
"""
Invoice OCR Processor - A comprehensive OCR tool for extracting information from invoices
"""

import os
import re
import json
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

# Suppress warnings
warnings.filterwarnings("ignore")

class InvoiceProcessor:
    """Main class for processing invoices using OCR technology"""
    
    def __init__(self, use_cv2: bool = True, use_keras_ocr: bool = False):
        """
        Initialize the Invoice Processor
        
        Args:
            use_cv2: Whether to use OpenCV for image processing
            use_keras_ocr: Whether to use Keras OCR for advanced text recognition
        """
        self.use_cv2 = use_cv2
        self.use_keras_ocr = use_keras_ocr
        
        # Try to import necessary libraries
        try:
            import pytesseract
            from PIL import Image
            import matplotlib.pyplot as plt
            self.pytesseract = pytesseract
            self.Image = Image
            self.plt = plt
            
            if use_cv2:
                try:
                    import cv2
                    self.cv2 = cv2
                    print("OpenCV imported successfully")
                except ImportError as e:
                    print(f"Error importing OpenCV: {e}")
                    print("Falling back to PIL for image processing")
                    self.use_cv2 = False
            
            if use_keras_ocr:
                try:
                    import keras_ocr
                    self.keras_ocr = keras_ocr
                    print("keras_ocr imported successfully")
                except ImportError as e:
                    print(f"Error importing keras_ocr: {e}")
                    self.use_keras_ocr = False
        
        except ImportError as e:
            print(f"Error importing required libraries: {e}")
            print("Please install missing dependencies")
            raise
    
    def load_image(self, image_path: str) -> Tuple[Any, Any]:
        """
        Load an image from the specified path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing the loaded image in different formats
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image with both PIL and OpenCV (if available)
        pil_img = self.Image.open(image_path)
        cv_img = None
        
        if self.use_cv2:
            cv_img = self.cv2.imread(image_path, self.cv2.IMREAD_COLOR)
        
        return cv_img, pil_img
    
    def display_image(self, image: Any, title: str = "Invoice Image") -> None:
        """
        Display an image using matplotlib
        """
        plt_img = image
        
        # Convert OpenCV image to RGB format for proper display
        if self.use_cv2 and isinstance(image, (self.cv2.UMat, self.cv2.Mat)):
            plt_img = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            
        self.plt.figure(figsize=(15, 10))
        self.plt.title(title)
        self.plt.imshow(plt_img)
        self.plt.axis('off')
        self.plt.show()
    
    def preprocess_image(self, image: Any) -> Any:
        """
        Preprocess the image to improve OCR accuracy
        """
        if not self.use_cv2:
            return image
        
        # Apply image processing techniques
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        blurred = self.cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = self.cv2.adaptiveThreshold(
            blurred, 255, 
            self.cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            self.cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def perform_ocr(self, image: Any, custom_config: str = '') -> str:
        """
        Perform OCR on the given image
        """
        # Default config for better results
        if not custom_config:
            custom_config = r'--oem 3 --psm 6'
        
        # Use pytesseract to extract text
        try:
            if self.use_cv2 and isinstance(image, (self.cv2.UMat, self.cv2.Mat)):
                # Process with OpenCV image
                preprocessed = self.preprocess_image(image)
                text = self.pytesseract.image_to_string(preprocessed, config=custom_config)
            else:
                # Process with PIL image
                text = self.pytesseract.image_to_string(image, config=custom_config)
            
            return text
        except Exception as e:
            print(f"Error performing OCR: {e}")
            return ""
    
    def perform_keras_ocr(self, image: Any) -> str:
        """
        Perform OCR using Keras OCR
        """
        if not self.use_keras_ocr:
            return ""
        
        try:
            # Initialize the keras-ocr pipeline
            pipeline = self.keras_ocr.pipeline.Pipeline()
            
            # Convert to RGB if using OpenCV image
            if self.use_cv2 and isinstance(image, (self.cv2.UMat, self.cv2.Mat)):
                image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            
            # Get predictions
            predictions = pipeline.recognize([image])[0]
            
            # Visualize predictions
            self.plt.figure(figsize=(10, 10))
            self.keras_ocr.tools.drawAnnotations(image=image, predictions=predictions)
            self.plt.show()
            
            # Extract text from predictions
            extracted_text = ' '.join([word[0] for word in predictions])
            return extracted_text
        
        except Exception as e:
            print(f"Error using keras_ocr: {e}")
            return ""
    
    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text
        """
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(email_pattern, text)
    
    def extract_amount(self, text: str) -> str:
        """
        Extract billing amount from text
        """
        # Look for context patterns
        billing_pattern = r'(?:total|amount|balance|due|invoice total)\s*[:\$]\s*([\$\d,.]+)'
        billing_matches = re.findall(billing_pattern, text.lower())
        
        if billing_matches:
            # Clean up the matched amount
            amount = re.search(r'\$[\d,]+\.\d{2}', billing_matches[0])
            if amount:
                return amount.group(0)
        
        # Fall back to simple dollar amount pattern
        amounts = re.findall(r'\$[\d,]+\.\d{2}', text)
        if amounts:
            return amounts[0]
        
        return "Not found"
    
    def extract_pattern(self, pattern: str, text: str, default: str = 'Not found') -> str:
        """
        Extract a pattern from text
        """
        match = re.search(pattern, text.lower())
        return match.group(1) if match else default
    
    def extract_all_info(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive information from invoice text
        """
        # Initialize dictionary to store all extracted information
        invoice_data = {
            'invoice_number': '',
            'date_of_issue': '',
            'date_due': '',
            'po_number': '',
            'region': '',
            'company_name': '',
            'company_address': [],
            'company_contact': '',
            'company_email': '',
            'customer_name': '',
            'customer_email': '',
            'total_amount': '',
            'items': []
        }
        
        # Define regex patterns for extracting information
        patterns = {
            'invoice_number': r'(?:invoice\s*number|invoice\s*#)\s*([\w-]+)',
            'date_of_issue': r'(?:date\s*of\s*issue|issued\s*on)\s*(\w+\s*\d{1,2},?\s*\d{4}|\d{1,2}[-/\s]\w+[-/\s]\d{4}|\w+\s*\d{1,2}[-/\s]\d{4})',
            'date_due': r'(?:date\s*due|due\s*date)\s*(\w+\s*\d{1,2},?\s*\d{4}|\d{1,2}[-/\s]\w+[-/\s]\d{4}|\w+\s*\d{1,2}[-/\s]\d{4})',
            'po_number': r'(?:po\s*number|purchase\s*order)\s*([\w-]+)',
            'region': r'region\s*(\w+)',
            'total_amount': r'\$(\d{1,3}(?:,\d{3})*\.\d{2})',
            'phone_number': r'(?:\+\d{1,4}\s*)?(?:\(\d{3,5}\)\s*)?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}'
        }
        
        # Process the text to extract all information
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Extract company name (usually at the top)
        if len(non_empty_lines) > 0:
            invoice_data['company_name'] = non_empty_lines[0]
        
        # Extract invoice number, dates, PO number, and region using regex
        invoice_data['invoice_number'] = self.extract_pattern(patterns['invoice_number'], text)
        invoice_data['date_of_issue'] = self.extract_pattern(patterns['date_of_issue'], text)
        invoice_data['date_due'] = self.extract_pattern(patterns['date_due'], text)
        invoice_data['po_number'] = self.extract_pattern(patterns['po_number'], text)
        invoice_data['region'] = self.extract_pattern(patterns['region'], text)
        invoice_data['total_amount'] = self.extract_pattern(patterns['total_amount'], text)
        
        # Extract company address (usually the lines after company name until contact info)
        address_started = False
        address_ended = False
        for line in non_empty_lines[1:10]:  # Check the first few lines only
            # Stop collecting address if we hit contact info or other sections
            if re.search(r'@|invoice\s*number|date|bill\s*to', line.lower()):
                address_ended = True
            
            if not address_ended and (re.search(r'\d+\s+[\w\s]+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|blvd|boulevard)', line.lower(), re.IGNORECASE) or 
                         re.search(r'[A-Z]{2}\s*\d{5}|[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}', line)):
                address_started = True
            
            if address_started and not address_ended:
                invoice_data['company_address'].append(line)
        
        # Extract phone number
        phone_match = re.search(patterns['phone_number'], text)
        if phone_match:
            invoice_data['company_contact'] = phone_match.group(0)
        
        # Extract emails
        emails = self.extract_emails(text)
        if emails:
            invoice_data['company_email'] = emails[0]
            # If there's a second email, it's likely the customer's
            if len(emails) > 1:
                invoice_data['customer_email'] = emails[1]
        
        # Extract customer name (usually after "Bill to")
        bill_to_index = -1
        for i, line in enumerate(non_empty_lines):
            if re.search(r'bill\s*to', line.lower()):
                bill_to_index = i
                break
        
        if bill_to_index != -1 and bill_to_index + 1 < len(non_empty_lines):
            # Customer name is usually the line after "Bill to"
            invoice_data['customer_name'] = non_empty_lines[bill_to_index + 1]
        
        # Try to extract line items
        items_section_start = -1
        for i, line in enumerate(non_empty_lines):
            if re.search(r'description\s+qty\s+unit\s*price\s+amount', line.lower()):
                items_section_start = i + 1
                break
        
        if items_section_start != -1:
            for line in non_empty_lines[items_section_start:]:
                # Stop if we hit a total line
                if re.search(r'total|due', line.lower()):
                    break
                    
                # Try to parse as item
                item_match = re.search(r'(.+?)\s+(\d+)\s+\$(\d[\d,.]+)\s+\$(\d[\d,.]+)', line)
                if item_match:
                    item = {
                        'description': item_match.group(1).strip(),
                        'quantity': item_match.group(2),
                        'unit_price': item_match.group(3),
                        'amount': item_match.group(4)
                    }
                    invoice_data['items'].append(item)
        
        return invoice_data
    
    def visualize_key_areas(self, image: Any, text_data: Dict) -> None:
        """
        Visualize key information areas in the image
        """
        if not self.use_cv2:
            print("Visualization skipped - OpenCV not available")
            return
        
        try:
            # Create a copy of the image for visualization
            vis_img = image.copy()
            
            # Use pytesseract to get bounding boxes for all text
            d = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            
            # Define keywords to highlight
            keywords = ['invoice', 'total', 'bill', 'due', 'date', 'number', 'email']
            
            # Draw rectangles around key information
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:  # Only consider text with high confidence
                    text = d['text'][i].lower()
                    if any(keyword in text for keyword in keywords) or '$' in text or '@' in text:
                        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                        self.cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display the image with highlighted text
            self.display_image(vis_img, "Key Information Highlighted")
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def export_to_json(self, data: Dict, output_file: str = 'extracted_invoice_data.json') -> None:
        """
        Export extracted data to JSON file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Extracted data exported to '{output_file}'")
        except Exception as e:
            print(f"Failed to export data: {e}")
    
    def process_invoice(self, image_path: str, visualize: bool = True, export: bool = True) -> Dict[str, Any]:
        """
        Process an invoice from start to finish
        """
        print(f"Processing invoice: {image_path}")
        
        # Load and display the image
        cv_img, pil_img = self.load_image(image_path)
        self.display_image(cv_img if self.use_cv2 and cv_img is not None else pil_img)
        
        # Perform OCR
        text = self.perform_ocr(cv_img if self.use_cv2 and cv_img is not None else pil_img)
        
        # Try Keras OCR if enabled
        if self.use_keras_ocr:
            keras_text = self.perform_keras_ocr(cv_img if self.use_cv2 and cv_img is not None else pil_img)
            if keras_text:
                print("Using Keras OCR results")
                text = keras_text
        
        # Extract all information
        invoice_data = self.extract_all_info(text)
        
        # Print the results
        self._print_invoice_data(invoice_data)
        
        # Visualize the results
        if visualize and self.use_cv2 and cv_img is not None:
            self.visualize_key_areas(cv_img, invoice_data)
        
        # Export the results
        if export:
            self.export_to_json(invoice_data)
        
        return invoice_data
    
    def _print_invoice_data(self, data: Dict[str, Any]) -> None:
        """
        Print the extracted invoice data in a structured format
        """
        print("\n===== INVOICE INFORMATION =====\n")
        print(f"Company: {data['company_name']}")
        print(f"Address: {' '.join(data['company_address'])}")
        print(f"Contact: {data['company_contact']}")
        print(f"Email: {data['company_email']}")
        
        print("\n----- INVOICE DETAILS -----")
        print(f"Invoice Number: {data['invoice_number']}")
        print(f"Date of Issue: {data['date_of_issue']}")
        print(f"Due Date: {data['date_due']}")
        print(f"PO Number: {data['po_number']}")
        print(f"Region: {data['region']}")
        
        print("\n----- CUSTOMER INFORMATION -----")
        print(f"Customer: {data['customer_name']}")
        print(f"Customer Email: {data['customer_email']}")
        
        print("\n----- FINANCIAL DETAILS -----")
        print(f"Total Amount: ${data['total_amount']}")
        
        if data['items']:
            print("\n----- LINE ITEMS -----")
            for idx, item in enumerate(data['items'], 1):
                print(f"Item {idx}:")
                print(f"  Description: {item['description']}")
                print(f"  Quantity: {item['quantity']}")
                print(f"  Unit Price: ${item['unit_price']}")
                print(f"  Amount: ${item['amount']}")


class BatchInvoiceProcessor:
    """Class for processing multiple invoices in batch mode"""
    
    def __init__(self, processor: InvoiceProcessor):
        """Initialize the batch processor"""
        self.processor = processor
        self.results = {}
    
    def process_directory(self, dir_path: str, extensions: List[str] = ['.jpg', '.jpeg', '.png', '.pdf']) -> Dict[str, Dict]:
        """Process all invoice images in a directory"""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a valid directory: {dir_path}")
        
        # Find all image files
        invoice_files = []
        for filename in os.listdir(dir_path):
            if any(filename.lower().endswith(ext) for ext in extensions):
                invoice_files.append(os.path.join(dir_path, filename))
        
        print(f"Found {len(invoice_files)} invoice files to process")
        
        # Process each file
        for file_path in invoice_files:
            filename = os.path.basename(file_path)
            print(f"\nProcessing {filename}...")
            try:
                # Process the invoice
                result = self.processor.process_invoice(
                    file_path, 
                    visualize=False,  # Don't visualize in batch mode
                    export=False  # Don't export individual results
                )
                self.results[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                self.results[filename] = {"error": str(e)}
        
        # Export the batch results
        self.export_results()
        
        return self.results
    
    def export_results(self, output_file: str = 'batch_invoice_results.json') -> None:
        """Export batch processing results to JSON"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"Batch results exported to '{output_file}'")
        except Exception as e:
            print(f"Failed to export batch results: {e}")


def main():
    """Main function to demonstrate the invoice OCR functionality"""
    
    print("Invoice OCR Processor")
    print("=====================")
    
    # Check if specific command line arguments were provided
    import sys
    if len(sys.argv) > 1:
        # Process a single invoice
        if os.path.isfile(sys.argv[1]):
            processor = InvoiceProcessor()
            processor.process_invoice(sys.argv[1])
        # Process a directory of invoices
        elif os.path.isdir(sys.argv[1]):
            processor = InvoiceProcessor()
            batch_processor = BatchInvoiceProcessor(processor)
            batch_processor.process_directory(sys.argv[1])
        else:
            print(f"Error: {sys.argv[1]} is not a valid file or directory")
    else:
        # Process the default invoice.jpg file
        if os.path.exists('invoice.jpg'):
            processor = InvoiceProcessor()
            processor.process_invoice('invoice.jpg')
        else:
            print("Error: No invoice file provided and 'invoice.jpg' does not exist")
            print("Usage: python invoice_ocr.py [invoice_file.jpg | directory_path]")


if __name__ == "__main__":
    main()