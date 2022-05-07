import firebase from "firebase/app"
import "firebase/auth"
import 'firebase/storage';

const app = firebase.initializeApp({
  apiKey: "AIzaSyCWg5SqMn4WzlsXAwAICOM4-1mP39S0V7w",
  authDomain: "hackfest-atlassian.firebaseapp.com",
  projectId: "hackfest-atlassian",
  storageBucket: "hackfest-atlassian.appspot.com",
  messagingSenderId: "92906433638",
  appId: "1:92906433638:web:d13c651a114f29f9f111a7",
  measurementId: "G-HGHEXLXQ0J"
})

const auth = app.auth()
const storage = app.storage()
export {storage, auth, app as default}
