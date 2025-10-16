import sys
import csv
import mglyph as mg
import os, random
from PyQt5 import QtCore, QtGui, QtWidgets
from io import BytesIO

from glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

SEED = None
random.seed(SEED)
N = 20 # pocet opakovani kazdeho glyphu

# ZDE MUZE BYT VOLBA MEZI SIMPLE A ADVANCED GLYPHS
USE_ADVANCED = True 
glyphs = {**ADVANCED_GLYPHS} if USE_ADVANCED else {**SIMPLE_GLYPHS}

def render_png(glyph_type: str, x: float) -> bytes:
    result = mg.render(glyphs[glyph_type], (96, 96), [x])
    pil_img = result[0]["pil"]
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# TODO: improve generating
# Generuj nahodne cislo x s uniformnim rozlozenim v rozmezi 10-50, to bude rozestup mezi A a C
# Generuj nahodne cislo 1 - 100-x, tam umisti A, C se umisti na A+x
# z = rand.int(0, 1) if z < 0.5 then A, C = C, A 
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.results = [] # pole pro ukladani csv vysledku 
        self.index = 1  # index pro pocitani vysledku
        self.glyph_index = 0

        self.glyph_order = []
        for glyph_type in glyphs.keys():
            self.glyph_order.extend([glyph_type] * N)
        random.shuffle(self.glyph_order)

        BINS = [(0, 35), (36, 70), (71, 100)]

        self.trials = []
        for glyph_type in self.glyph_order:
            while True:
                x = random.randint(10, 50)
                idx = random.randint(0, 2)
                bin_low, bin_high = BINS[idx]

                c_low = max(bin_low, x + 1)
                c_high = min(bin_high, 100)

                if c_low <= c_high:
                    c = random.randint(c_low, c_high)
                    a = c - x
                    break

            if random.random() < 0.5:
                a, c = c, a

            self.trials.append((glyph_type, a, c))
            
        # _____________GUI setup______________________
        self.setWindowTitle("Experiment")
        self.setGeometry(400, 300, 900, 500)

        main_layout = QtWidgets.QVBoxLayout(self)
        glyph_layout = QtWidgets.QHBoxLayout()

        first_glyph, a0, c0 = self.trials[0]

        self.glyphA = GlyphWidget(first_glyph, a0, editable = False)
        glyph_layout.addWidget(self.glyphA)

        self.glyphB = GlyphWidget(first_glyph, 1, editable = True)
        glyph_layout.addWidget(self.glyphB)

        self.glyphC = GlyphWidget(first_glyph, c0, editable = False)
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
        if hasattr(self, 'sizeA') and hasattr(self, 'sizeC'):
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
            QtWidgets.QMessageBox.information(self, "Done", "Results saved to data/data_sets/results.csv")
            self.close()
            return

        # pokracovani experimentu
        glyph_type, self.sizeA, self.sizeC = self.trials[self.glyph_index]
        self.current_glyph_type = glyph_type
        self.glyph_index += 1

        # nastaveni stejneho glyphu pro trojici glyphu
        self.glyphA.set_type(glyph_type)
        self.glyphB.set_type(glyph_type)
        self.glyphC.set_type(glyph_type)

        # nastaveni pevne velikosti pro glyphy A a C
        self.glyphA.set_value(self.sizeA)
        self.glyphC.set_value(self.sizeC)

        # setup velikosti pro glyph B na 1
        self.glyphB.set_value(1)

        self.update_counter_label()

    def save_results(self, filename="data/data_sets/results.csv"):
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