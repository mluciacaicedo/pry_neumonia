# src/controller.py

import os
from integrator import run_pipeline
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import csv

class PneumoniaController:
    def __init__(self):
        self.reportID = 0

    def predict_image(self, filepath):
        """
        Ejecuta la predicción sobre la imagen usando el pipeline existente.
        Devuelve: label, probabilidad, heatmap
        """
        label, proba, heatmap = run_pipeline(filepath)
        return label, proba, heatmap

    def save_pdf(self, patient_id, label, proba, heatmap):
        """
        Genera un PDF con la información del paciente y la imagen del heatmap.
        """
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        overlay_path = os.path.join(reports_dir, f"overlay_{self.reportID}.png")
        Image.fromarray(heatmap).save(overlay_path)

        pdf_path = os.path.join(reports_dir, f"Reporte{self.reportID}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Reporte de Detección de Neumonía")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Cédula Paciente: {patient_id}")
        c.drawString(50, height - 120, f"Resultado: {label}")
        c.drawString(50, height - 140, f"Probabilidad: {proba:.2f}%")
        c.drawImage(overlay_path, 50, 200, 400, 400)

        c.save()
        self.reportID += 1
        return pdf_path

    def save_csv(self, patient_id, label, proba):
        """
        Guarda la predicción en un CSV de historial
        """
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        csv_path = os.path.join(reports_dir, "historial.csv")
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([patient_id, label, f"{proba:.2f}%"])
        return csv_path




