import React from "react";
import styled from "styled-components";
import { Link } from "react-router-dom";
import "./styles.css";

function Home() {
  const Container = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("/background_1.png");
    background-size: cover;
  `;
  return (
    <Container>
      <p className="title wobble animated">
        <img src="/title.png" />
      </p>
      <p className="subtitle">
        <Link to="/select">
          <img src="/subtitle.png" />
        </Link>
      </p>
    </Container>
  );
}

export default Home;
