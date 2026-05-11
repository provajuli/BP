import qrcode

qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=1,
)

qr.add_data("https://www.stud.fit.vutbr.cz/~xprova06/#/")
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("qr.png")