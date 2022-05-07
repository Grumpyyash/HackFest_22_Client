import React from "react"
import { Card } from "react-bootstrap"

export default function Banner() {
  return(
    <div className="d-flex justify-content-center">
      <Card
        bg="info"
        text="white"
        style={{ width: '18rem' }}
        className="m-3"
      >
        <Card.Body>
          <Card.Title> 102 STUDENTS </Card.Title>
        </Card.Body>
      </Card>
      <Card
        bg="success"
        text="white"
        style={{ width: '18rem' }}
        className="m-3"
      >
        <Card.Body>
          <Card.Title> 91 PRESENT </Card.Title>
        </Card.Body>
      </Card>
      <Card
        bg="danger"
        text="white"
        style={{ width: '18rem' }}
        className="m-3"
      >
        <Card.Body>
          <Card.Title> 11 ABSENT </Card.Title>
        </Card.Body>
      </Card>
    </div>
  );
}