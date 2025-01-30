import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QScrollArea, QGridLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image
import os

from cell_alignment import image_processing

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Selected images layout
        self.selected_images_label = QLabel("No images selected", self)
        self.selected_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.selected_images_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # Output images layout
        self.output_scroll_area = QScrollArea(self)
        self.output_scroll_area.setWidgetResizable(True)
        self.output_content = QWidget()
        self.output_layout = QHBoxLayout(self.output_content)
        self.output_scroll_area.setWidget(self.output_content)
        layout.addWidget(self.output_scroll_area)

        self.select_button = QPushButton("Select Images", self)
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        self.process_button = QPushButton("Process Images", self)
        self.process_button.clicked.connect(self.process_images)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)

        self.setLayout(layout)
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 800, 600)
        self.selected_images = []
        self.save_directory = None

    def open_file_dialog(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        
        if file_names:
            self.selected_images = file_names
            self.selected_images_label.setText(f"{len(file_names)} images selected")
            self.display_selected_images()
            self.process_button.setEnabled(True)

    def display_selected_images(self):
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)

        for i, path in enumerate(self.selected_images):
            label = QLabel(self)
            pixmap = QPixmap(path)
            label.setPixmap(pixmap.scaled(self.scroll_area.width() // 4, self.scroll_area.width() // 4, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.scroll_layout.addWidget(label, i // 4, i % 4)

    def save_image(self, index):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if save_path:
            if not any(save_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]):
                save_path += ".png"
                
            # Assuming the last processed image is saved
            if self.output_pixmaps:
                self.output_pixmaps[index].save(save_path)

    def process_images(self):
        # Clear old output images
        for i in reversed(range(self.output_layout.count())):
            self.output_layout.itemAt(i).widget().setParent(None)

        self.output_pixmaps = image_processing(self.selected_images)  # Process images

        for idx, pixmap in enumerate(self.output_pixmaps):
            hbox = QHBoxLayout()

            # Display image
            label = QLabel(self)
            label.setPixmap(pixmap)
            hbox.addWidget(label)

            # Save button
            save_button = QPushButton("Save Image")
            save_button.clicked.connect(lambda checked, index=idx: self.save_image(index))
            hbox.addWidget(save_button)

            self.output_layout.addLayout(hbox)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
