from flask import Flask,request
from flask import render_template
from coin_triming_canny import Triming
import zipfile
app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/views",methods=['POST'])
def view():
    print("OK")
    a = request.form["image"]
    print(a)
    return render_template('view.html')

@app.route("/upload",methods=['POST'])
def upload():
    file_list = request.files.getlist("image")
    zip_name = "./static/zip_files/"+"data.zip"
    zipFile = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    path_list = []
    for file in file_list:
        print(file.filename)
        # ここでいったん保存
        file.save("./static/upload_file/" + file.filename)
        trimg_class = Triming(file.filename)
        trimg_class.run_canny()
        paths = './static/triminged/' + file.filename
        path_list.append(paths)
        zipFile.write("./static/triminged/"+file.filename)
    zipFile.close()
    return render_template('fin.html',img_paths=path_list,zip_name=zip_name)

if __name__ == "__main__":
    app.run()