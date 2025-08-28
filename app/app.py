import sys
import csv
import math
import random
import colorsys

from PyQt5 import QtCore, QtGui, QtWidgets
from io import BytesIO
import mglyph as mg

# Registrace glyphu
def horizontal_line(x: float, canvas:mg.Canvas) -> None:
    canvas.line((mg.lerp(x, canvas.xcenter, canvas.xleft), canvas.ycenter),    
                (mg.lerp(x, canvas.xcenter, canvas.xright), canvas.ycenter),   
                linecap='round', width='30p', color='navy')

def simple_scaled_square(x: float, canvas: mg.Canvas) -> None:
    tl = (mg.lerp(x, 0.0, -1), mg.lerp(x, 0.0, -1.0))
    br = (mg.lerp(x, 0, 1), mg.lerp(x, 0, 1))

    canvas.rect(tl, br, color='blue', width='20p')

def simple_scaled_circle(x: float, canvas: mg.Canvas) -> None:
    canvas.circle(canvas.center, mg.lerp(x, 0, 1), color='red', style='stroke', width='20p')

def simple_scaled_star(x: float, canvas: mg.Canvas) -> None:
    radius = mg.lerp(x, 0, canvas.ysize/2)
    vertices = []
    for segment in range(5):
        vertices.append(mg.orbit(canvas.center, segment * 2*math.pi/5, radius))
        vertices.append(mg.orbit(canvas.center, (segment + 0.5) * 2*math.pi/5, math.cos(2*math.pi/5)/math.cos(math.pi/5) * radius))
    canvas.polygon(vertices, color=(1,0,0,0.25), closed=True, width='30p')

def simple_polygon(x: float, canvas: mg.Canvas) -> None:
    l = (mg.lerp(x, canvas.xcenter, canvas.xleft), canvas.ycenter)
    ct = (canvas.xcenter, -0.2)
    r = (mg.lerp(x, canvas.xcenter, canvas.xright), canvas.ycenter)
    cb = (canvas.xcenter, 0.1)
    canvas.polygon([l, cb, r, ct], color='#A2a', closed=True)

def simple_colour_patch(x: float, canvas: mg.Canvas) -> None:
    hue = x / 100
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    canvas.rect(
        (canvas.xleft, canvas.ytop),
        (canvas.xright, canvas.ybottom),
        color=color,
        style="fill",
    )

glyphs = {
    "line": horizontal_line,
    "square": simple_scaled_square,
    "circle": simple_scaled_circle,
    "star": simple_scaled_star,
    "polygon": simple_polygon,
}

def render_png(glyph_type: str, x: float) -> bytes:
    result = mg.render(glyphs[glyph_type], (96, 96), [x])
    pil_img = result[0]["pil"]
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.results = [] # pole pro ukladani csv vysledku 
        self.index = 1 # csv index
        self.glyph_order = [] # serazeni typu glyphu 
        for glyph_type in glyphs:
            self.glyph_order.extend([glyph_type] * 20)
        random.shuffle(self.glyph_order)
        self.glyph_index = 0 # pro pocitadlo

        self.setWindowTitle("Experiment")
        self.setGeometry(400, 300, 900, 500)

        main_layout = QtWidgets.QVBoxLayout(self)
        glyph_layout = QtWidgets.QHBoxLayout()

        self.glyphA = GlyphWidget("line", 0, editable = False)
        glyph_layout.addWidget(self.glyphA)

        self.glyphB = GlyphWidget("line", 50, editable = True)
        glyph_layout.addWidget(self.glyphB)

        self.glyphC = GlyphWidget("line", 100, editable = False)
        glyph_layout.addWidget(self.glyphC)

        self.counter_label = QtWidgets.QLabel()
        self.counter_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.counter_label.setFont(font)
        main_layout.addWidget(self.counter_label)

        glyph_wrapper = QtWidgets.QHBoxLayout()
        glyph_wrapper.addStretch()
        glyph_wrapper.addLayout(glyph_layout)
        glyph_wrapper.addStretch()
        main_layout.addLayout(glyph_wrapper)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_next.clicked.connect(self.new_example)
        
        button_layout.addStretch()
        button_layout.addWidget(self.btn_next)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.new_example()

    def new_example(self):
        #ulozeni vysledku 
        if hasattr(self, 'sizeA') and hasattr(self, 'sizeB') and hasattr(self, 'sizeC'):
            result = {
                'index': self.index,
                'glyph_type': self.current_glyph_type,
                'sizeA': self.sizeA,
                'sizeB': self.glyphB.value,
                'sizeC': self.sizeC,
            }
            self.results.append(result)
            self.index += 1
        
        # ukonceni experimentu
        if self.glyph_index >= len(self.glyph_order):
            QtWidgets.QMessageBox.information(self, "Done", "Results saved to data/results.csv")
            self.close()
            return

        # pokracovani experimentu
        glyph_type = self.glyph_order[self.glyph_index]
        self.current_glyph_type = glyph_type
        self.glyph_index += 1

        # nastaveni stejneho glyphu pro trojici glyphu
        self.glyphA.set_type(glyph_type)
        self.glyphB.set_type(glyph_type)
        self.glyphC.set_type(glyph_type)

        # generovani nahodnych velikosti pro glyphy A a C
        while True:
            self.sizeA = random.randint(1, 100)
            self.sizeC = random.randint(1, 100)
            if (self.sizeA < self.sizeC) or (self.sizeA > self.sizeC):
                break

        # nastaveni pevne velikosti pro glyphy A a C
        self.glyphA.set_value(self.sizeA)
        self.glyphC.set_value(self.sizeC)

        # setup velikosti pro glyph B na prumer velikosti glyphu A a C
        self.sizeB = 1
        self.glyphB.set_value(self.sizeB)

        self.update_counter_label()

    def save_results(self, filename="data/results.csv"):
        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["index", "glyph_type", "sizeA", "sizeB", "sizeC"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

    def update_counter_label(self):
        total = len(self.glyph_order)
        self.counter_label.setText(f"{self.index} / {total}")

class GlyphWidget(QtWidgets.QWidget):
    def __init__(self, glyph_type: str, value: float, editable: bool = False):
        super().__init__()

        self.glyph_type = glyph_type
        self.value = value
        self.editable = editable

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)

        self.image_label = QtWidgets.QLabel()
        layout.addWidget(self.image_label)

        #self.value_label = QtWidgets.QLabel() # DEBUG
        #self.value_label.setAlignment(QtCore.Qt.AlignCenter) # DEBUG
        #layout.addWidget(self.value_label) # DEBUG

        self.update_image()

    def set_value(self, value: float):
        self.value = max(0, min(100, value))
        self.update_image()

    def set_type(self, glyph_type: str):
        if glyph_type in glyphs:
            self.glyph_type = glyph_type
            self.update_image()
        else:
            raise ValueError(f"Unknown glyph type: {glyph_type}")

    def update_image(self):
        image = render_png(self.glyph_type, self.value)
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(image)
        self.image_label.setPixmap(pixmap)

        #self.value_label.setText(f"DEBUG: {self.value:.1f}") # DEBUG

    def wheelEvent(self, event):
        if self.editable:
            step = 1
            delta = step if event.angleDelta().y() > 0 else -step
            self.set_value(min(100, max(0, self.value + delta)))
            self.update_image()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
exit_code = app.exec_()
window.save_results() 
sys.exit(exit_code)