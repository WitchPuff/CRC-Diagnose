<!-- eslint-disable vue/no-mutating-props -->
<template>
  <card :style="{ textAlign: 'center' }">
    <h3 slot="header" class="title">AI辅助直结肠癌诊断
      <!-- <h4>via Image Upload</h4> -->
    </h3>
    <div class="row">
          <div class="upload-box" @click="$refs.fileInput.click()" >
            <i class="fas fa-plus"></i>
            <input type="file" ref="fileInput" style="display:none" @change="uploadImage" accept="image/*">
          </div>
    </div> 

    <h3 :style="{ marginTop: '2%' }">预测标签: {{ prediction }}</h3>
    <h3 :style="{ marginTop: '-1%' }">AI诊断意见: {{ note }}</h3>
    <h4 :style="{ marginTop: '-1%' }">模型选择:
      <select v-model="model" style="margin-left: 10px; height: 1.5em;">
        <option value="">--model--</option>
        <option value="resnet">ResNet</option>
        <option value="vit">VIT</option>
      </select>
      <base-button 
        :style="{ 
          marginLeft: '2%', 
          padding: '10px 10px', // adjust as needed
          boxSizing: 'border-box'
           
        }" 
        slot="footer" 
        type="primary" 
        fill 
        @click="voice4model()"
      >
        声音控制
      </base-button>
      <br>
      <span :style="{marginLeft: '1%', fontSize: '0.7em'}">{{ state }}</span>
    </h4>
    <br>
    <h4 :style="{ marginTop: '-2%' }">{{ message }}</h4>
    <base-button :style="{ marginTop: '-2%', fontSize: '1.3em', lineHeight: '1.2em' }" slot="footer" type="primary" fill @click="predict()">Predict</base-button>

  </card>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      imageUrl: null,
      img: null,
      model: null,
      prediction: null,
      state: null,
      message: null,
      note: null
    };
  },
  methods: {
    uploadImage(event) {
      const formData = new FormData();
      formData.append('file', event.target.files[0]);
      this.model = null;
      this.prediction = null;
      axios.post('http://localhost:5000/api/upload', formData)
      .then(response => {
        this.imageUrl = 'http://localhost:5000/api/upload/' + response.data.filename;
        this.img = response.data.filename;
        document.querySelector('.upload-box').style.backgroundImage = `url(${this.imageUrl})`;
        document.querySelector('.upload-box').style.backgroundSize = 'cover';
        document.querySelector('.upload-box i').style.display = 'none';
        document.querySelector('.upload-box').style.border = 'none';
      })
      .catch(error => console.error(error));
    },
    predict() {
      if (this.img && this.model) {
        this.message = 'Diagnosing...';
        axios.get('http://localhost:5000/api/predict/' + this.model + '/' + this.img)
        .then(response => {
          // 这里处理服务器的响应
          this.message = null;
          this.prediction = response.data.result;
          // 根据不同的 prediction 设置相应的 note 和完整名词
          switch (this.prediction) {
            case 'ADI':
              this.note = '脂肪组织 - 正常脂肪组织，未见异常。';
              this.prediction = 'Adipose';
              break;
            case 'BACK':
              this.note = '背景组织 - 组织切片的基础结构，无异常发现。';
              this.prediction = 'Background';
              break;
            case 'DEB':
              this.note = '碎片 - 可能是由于组织制备过程中的破损，未显示任何特定病理变化。';
              this.prediction = 'Debris';
              break;
            case 'LYM':
              this.note = '淋巴细胞 - 表现出典型的淋巴细胞浸润，可能是炎症或免疫反应的结果。';
              this.prediction = 'Lymphocytes';
              break;
            case 'MUC':
              this.note = '黏液 - 存在黏液积聚，可能与某些疾病状态有关，例如炎症。';
              this.prediction = 'Mucus';
              break;
            case 'MUS':
              this.note = '平滑肌组织 - 正常的平滑肌组织，未见异常。';
              this.prediction = 'Smooth Muscle';
              break;
            case 'NORM':
              this.note = '正常结肠黏膜 - 组织呈正常结构，未见明显的病理变化。';
              this.prediction = 'Normal Colon Mucosa';
              break;
            case 'STR':
              this.note = '癌相关基质 - 存在与癌症关联的纤维组织，可能反映了肿瘤周围的反应性改变。';
              this.prediction = 'Cancer-Associated Stroma';
              break;
            case 'TUM':
              this.note = '结直肠腺癌上皮 - 存在结直肠腺癌的上皮组织，可能包括异型细胞和细胞增生。';
              this.prediction = 'Colorectal Adenocarcinoma Epithelium';
              break;
            default:
              // 默认情况，保持原始值
              this.note = null;
          }
          this.speakPrediction(this.prediction);
          console.log(response.data.result);
        })
        .catch(error => console.error(error));
      }
      else {
        this.message = 'Upload a image / choose a model first.';
      }
  },
  speakPrediction(prediction) {
  // Check if the SpeechSynthesis API is available
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(prediction);

    // Optionally, you can set voice and other properties here
    // utterance.voice = ...

    // Speak the prediction
    speechSynthesis.speak(utterance);
  } else {
    console.log('SpeechSynthesis not supported');
  }
},
  voice4model() {
    // Check if the browser supports the SpeechRecognition API
    if ('webkitSpeechRecognition' in window) {
      const recognition = new webkitSpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.start();
      console.log('started');
      this.state = '开始录音，请说one/two来选择模型ResNet/ViT。'
      recognition.onresult = (event) => {
        // Get the spoken text
        this.state = '正在转录...'
        const speechResult = event.results[0][0].transcript[0];
        console.log(speechResult);
        this.state = speechResult;
        // Check what was said and select the appropriate model
        if (speechResult === '1') {
          this.model = 'resnet';
          console.log('resnet');
          recognition.stop();
        } else if (speechResult === '2') {
          this.model = 'vit';
          console.log('vit');
          recognition.stop();
        }
        // Stop the recognition
        recognition.stop();
        console.log('stopped');
        this.state = null;
      };

      recognition.onerror = (event) => {
        // Handle any errors
        recognition.stop();
        console.error('Error occurred in recognition:', event.error);
        console.log('stopped');
        this.state = "未识别到语音，录音暂停。";
      };
    } else {
      console.error('SpeechRecognition API is not supported in this browser.');
    }
},


  },
};
</script>
<style>


.upload-box {
  width: 200px;
  height: 200px;
  border: 2px dashed #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin-left: auto; /* 尝试添加这行 */
  margin-right: auto; /* 尝试添加这行 */
}
.upload-box i {
  font-size: 50px;
}
</style>
