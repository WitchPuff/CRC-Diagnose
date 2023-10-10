<!-- eslint-disable vue/no-mutating-props -->
<template>
  <card :style="{ textAlign: 'center' }">
    <h3 slot="header" class="title">AI-Powered CRC Diagnosis
      <h4>via Image Upload</h4>
    </h3>
    <div class="row">
          <div class="upload-box" @click="$refs.fileInput.click()" >
            <i class="fas fa-plus"></i>
            <input type="file" ref="fileInput" style="display:none" @change="uploadImage" accept="image/*">
          </div>
    </div> 

    <h3 :style="{ marginTop: '2%' }">Label: {{ prediction }}</h3>
    <h4 :style="{ marginTop: '-2%' }">Model:
      <select v-model="model" style="margin-left: 10px; height: 1.5em;">
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
        Voice Control
      </base-button>
      <br>
      <span :style="{marginLeft: '1%', fontSize: '0.7em'}">{{ state }}</span>
    </h4>
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
      state: null
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
    axios.get('http://localhost:5000/api/predict/' + this.model + '/' + this.img)
    .then(response => {
      // 这里处理服务器的响应
      this.prediction = response.data.result;
      console.log(response.data.result);
    })
    .catch(error => console.error(error));
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
    this.state = 'Recording started. Say 1/2 to choose the model for diagnosis.'
    recognition.onresult = (event) => {
      // Get the spoken text
      this.state = 'Transcribing...'
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
      this.state = "Hearing nothing, recording stopped.";
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
