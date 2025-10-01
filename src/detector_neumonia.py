# detector_neumonia.py
"""
Parte grafica en en Tkinter para el diagnostico de neumonia a partir de imagenes radiograficas
-Carga de imagenes en formato  DICOM, JPG, JPEG, PNG
-Preprocesa la imagen y ejecuta el modelo de prediccion
-Muestra el diagnostico con la probabildiad asociada
-Genera el reporte en PDF y guarda el historial en CSV dentro de la carpeta reports
"""
import os

from tkinter import *
from tkinter import ttk, filedialog, font 
from tkinter.messagebox import askokcancel, showinfo, WARNING
from  read_img import read_dicom_file, read_jpg_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageTk
from integrator import run_pipeline
import csv
import tkcap
import cv2
from controller.controller import PneumoniaController

class App:
    def __init__(self):
        self.root = Tk()
        self.controller = PneumoniaController()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None
        self.filepath = None  # guardará el archivo cargado
        self.label = ""
        self.proba = 0.0
        self.heatmap = None

        #  NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )
        if filepath:
            self.filepath = filepath
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".dcm":
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)

            img2show = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(img2show)
            self.text_img1.delete("1.0", "end")
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"


    def run_model(self):
        result_label, proba, heatmap = self.controller.predict_image(self.filepath)
        self.label = result_label
        self.proba = proba
        self.heatmap = heatmap

        img2 = Image.fromarray(self.heatmap)
        img2 = img2.resize((250, 250), Image.Resampling.LANCZOS)

        self.img2_tk = ImageTk.PhotoImage(img2)
        self.text_img2.delete("1.0", "end")
        self.text_img2.image_create("end", image=self.img2_tk)

        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")
        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}%")
       
    def create_pdf(self):
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        overlay_img = getattr(self, "saved_overlay_path", None)
        print("heatmap:", self.heatmap)

        if self.label is None or self.heatmap is None: 
            from tkinter.messagebox import showinfo
            showinfo("Error", "Primero debes generar una predicción.")
            return
        pdf_path = os.path.join(reports_dir, f"Reporte{self.reportID}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
    
        c.drawString(50, height - 50, "Reporte de Detección de Neumonía")
        # Datos del paciente 
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Cédula Paciente: {self.text1.get()}")
        c.drawString(50, height - 120, f"Resultado: {self.label}")
        c.drawString(50, height - 140, f"Probabilidad: {self.proba:.2f}%")
        # Imagen de la predicción
        img_temp_path = os.path.join(reports_dir, "temp_imag.png")
        img = Image.fromarray(self.heatmap)
        img.save(img_temp_path)

        c.drawImage(img_temp_path, 50, height - 550, width=400, height=400)
        # Guardar PDF
        c.save()
        self.reportID += 1
        from tkinter.messagebox import showinfo
        showinfo("PDF", f"El PDF fue generado con éxito:\n{pdf_path}")
        os.remove(img_temp_path)

        
    def save_results_csv(self):
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        csv_path = os.path.join(reports_dir, "historial.csv")
        with open(csv_path, "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow( [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"] )
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            self.array = None
            self.button1["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")        

if __name__ == "__main__":
    app = App()