import React, { useState, useEffect } from "react"
import {storage} from '../firebase';
import 'firebase/storage';
import { Card, Button, Alert } from "react-bootstrap"
import { useAuth } from "../contexts/AuthContext"
import { Link, useHistory } from "react-router-dom"

export default function Dashboard() {
  const [files, setFiles] = useState();

  useEffect(() => {
      const fetchImages = async () => {
        let result = await storage.ref().child("images").listAll();
        console.log(result);
        let urlPromises = result.items.map((imageRef) =>
          imageRef.getDownloadURL()
        );

        return Promise.all(urlPromises);
      };

      const loadImages = async () => {
        const urls = await fetchImages();
        setFiles(urls);
        console.log(urls);
      };
      loadImages();
  }, []);

  const [error, setError] = useState("")
  const { currentUser, logout } = useAuth()
  const history = useHistory()

  async function handleLogout() {
    setError("")

    try {
      await logout()
      history.push("/login")
    } catch {
      setError("Failed to log out")
    }
  }

  const [file, setFile] = useState(null);
  const [url, setURL] = useState("");

  function handleChange(e) {
    if (e.target.files[0])
        setFile(e.target.files[0]);
  }

  async function handleUpload(e){
    e.preventDefault();
    const path = `/images/${file.name}`;
    const ref = storage.ref(path);
    await ref.put(file);
    const url = await ref.getDownloadURL()
    setURL(url);
    setFile(null);
  }

  return (
    <>
      <div>
        <form onSubmit={handleUpload}>
          <input type="file" onChange={handleChange} />
          <button disabled={!file}>upload to firebase</button>
        </form>
      </div>
      <Card>
        <Card.Body>
          <h2 className="text-center mb-4">Profile</h2>
          {error && <Alert variant="danger">{error}</Alert>}
          <strong>Email:</strong> {currentUser.email}
          <Link to="/update-profile" className="btn btn-primary w-100 mt-3">
            Update Profile
          </Link>
        </Card.Body>
      </Card>
      <div className="w-100 text-center mt-2">
        <Button variant="link" onClick={handleLogout}>
          Log Out
        </Button>
      </div>
      <div style={{margin: "auto", textAlign: "center"}}>
        {files ? files.map((img) => (
          <img src={img} style={{maxWidth: "400px", textAlign: "center", margin: "auto"}} />
        )) : null}
      </div>
    </>
  )
}
