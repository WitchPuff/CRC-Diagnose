<template>
    <div class="login-container">
      <div class="login-content">
        <h2>Login</h2>
        <form @submit.prevent="login">
          <div class="form-group">
            <label for="username">Username:</label>
            <input type="text" v-model="username" required />
          </div>
          <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" v-model="password" required />
          </div>
          <!-- Use base-button for the login button -->
          <base-button type="primary" fill @click="login()">Login</base-button>
        </form>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        username: "",
        password: "",
      };
    },
    methods: {
      login() {
        // Simulate a login process (replace with your actual login logic)
        console.log("Logging in with "+ this.username + " " + this.password)
        if (this.username === "demo" && this.password === "test") {
          // Redirect to the dashboard upon successful login
          this.$router.push("/content");
        } else {
          alert("Invalid username or password");
        }
      },
    },
  };
  </script>
  
  <style scoped>
  .login-container {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh; /* Ensure full height of the viewport */
  }
  
  .login-content {
    max-width: 400px;
    width: 100%;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }
  
  .form-group {
    margin-bottom: 15px;
  }
  
  label {
    display: block;
    margin-bottom: 5px;
    font-size: large;
  }
  
  input {
    width: 100%;
    padding: 8px;
    box-sizing: border-box;
  }
  </style>
  