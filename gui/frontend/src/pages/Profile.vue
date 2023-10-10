<template>
  <div class="row">
    <div class="col-md-12">
      <edit-profile-form :model="model">
      </edit-profile-form>
    </div>
    <!-- <div class="col-md-4">
      <user-card :user="user"></user-card>
    </div> -->
    
  </div>
</template>
<script>
  import EditProfileForm from './Profile/EditProfileForm';
  export default {
    components: {
      EditProfileForm
    },
    data() {
      return {
        model: {

        }
      }
    },
    methods: {

    }
  }
</script>
<style>
</style>
