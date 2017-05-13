from flask import Flask,request
from flask import render_template
from coin_triming_canny import Triming
import zipfile
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload",methods=['POST'])
def upload():
    the_file = request.files["filename"]
    # ここでいったん保存
    the_file.save("./static/upload_file/" + the_file.filename)
    trimg_class = Triming(the_file.filename)
    trimg_class.run_canny()
    zip_name = "./static/zip_files/"+the_file.filename+".zip"
    zipFile = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zipFile.write("./static/triminged/"+the_file.filename)
    zipFile.close()
    return render_template('fin.html',img_path='./static/triminged/'+the_file.filename,zip_name=zip_name)

if __name__ == "__main__":
    app.run()