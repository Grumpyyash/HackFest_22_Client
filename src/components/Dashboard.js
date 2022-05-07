import React, { useState, useEffect } from "react"
import {storage} from '../firebase';
import 'firebase/storage';
import { Card, Button, Alert, Container, Form } from "react-bootstrap"
import { useHistory } from "react-router-dom"

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
  const history = useHistory()


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

      <Container
        className="d-flex align-items-center justify-content-center"
        style={{ minHeight: "100vh" }}
      >
      <div className="w-100" style={{ maxWidth: "400px" }}>
      <Card>
        <Card.Body>
          <h2 className="text-center mb-4">Upload Photo</h2>
          {error && <Alert variant="danger">{error}</Alert>}
          <Form onSubmit={handleUpload}>
            <input type="file" onChange={handleChange} />
            <Button disabled={!file} className="w-100" type="submit">
              Upload to Firebase
            </Button>
          </Form>
        </Card.Body>
      </Card>
      </div>
    </Container>
      <div style={{margin: "auto", textAlign: "center"}}>
        {files ? files.map((img) => (
          <img src={img} style={{maxWidth: "400px", textAlign: "center", margin: "auto"}} />
        )) : null}
      </div>
    </>
  )
}
