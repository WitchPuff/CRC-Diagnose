from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import *
from utils import predict as predictImg
from PIL import Image
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})


# 存储上传文件的目录
current_file_path = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_file_path, 'upload')
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # 打开图片文件
        img = Image.open(filename)
        # 将图片转换为JPG格式
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        img.save(jpg_filename, 'JPEG')

        # 返回文件的URL
        file_url = jpg_filename.replace('\\','/')
        print(file_url)
        return jsonify({'filename': file.filename[:-3] + "jpg"})

@app.route('/api/upload/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/predict/<path:model>/<path:filename>')
def predict(model, filename):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if model.lower() == 'resnet34' or model.lower() == 'resnet':
        net = ResNet(BasicBlock, (16,32,64,128), (3,4,6,3), num_classes=9, type=34)
        pt = os.path.join(current_file_path, "../../models/ResNet34_CRC.pt")
    # elif model.lower() == 'resnet50':
    #     net = ResNet(BottleNeck, (16,32,64,128), (3,4,6,3), num_classes=9, type=50)
    #     pt = ""
    elif model.lower() == 'vit':
        net = ViT(num_classes=9)
        pt = os.path.join(current_file_path, "../../models/ViT_CRC.pt")
    else:
        return "ERROR: No Such Model", 400
    label = predictImg(model=model, net=net, pt=pt, dataset='CRC-VAL-HE-7K', img=filename)
    if not label[0]:
        return label[1], 400
    else:
        return jsonify({'result': label})

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:fallback>')
def fallback(fallback):       # Vue Router 的 mode 为 'hash' 时可移除该方法
    if fallback.startswith('css/') or fallback.startswith('js/')\
            or fallback.startswith('img/') or fallback == 'favicon.ico':
        return app.send_static_file(fallback)
    else:
        return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=DEBUG)
