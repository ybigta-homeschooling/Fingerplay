import React, { useState, useEffect, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import styled from "styled-components";
import APIService from "../components/APIService";
import CamVideo from "../components/Camvideo";

function Cam() {
  const { animalId, level } = useParams();
  const [score, setScore] = useState(-1);
  const [wrongList, setWrongList] = useState([]);
  const [finish, setFinish] = useState(false);
  const Container = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("/background_3.png");
    background-size: cover;
  `;
  const refreshPage = () => {
    window.location.reload();
  };
  const getResult = () => {
    fetch(`http://localhost:5003/video_feed/${animalId}/${level}/fin`)
      .then((res) => {
        if (res.statue != 200) {
        }
        return res.json();
      })
      .then((data) => {
        setScore(data["score"]);
        setWrongList(data["wa"]);
      });
  };
  useEffect(() => {
    //   const timer = setInterval(getResult, 10000);
    //   return () => {
    //     clearInterval(timer);
    //   };
    setTimeout(getResult, 40000);
  }, []);

  const AsyncImage = (props) => {
    const [loadedSrc, setLoadedSrc] = React.useState(null);
    React.useEffect(() => {
      setLoadedSrc(null);
      if (props.src) {
        const handleLoad = () => {
          setLoadedSrc(props.src);
        };
        const image = new Image();
        image.addEventListener("load", handleLoad);
        image.src = props.src;
        return () => {
          image.removeEventListener("load", handleLoad);
        };
      }
    }, [props.src]);
    if (loadedSrc === props.src) {
      return <img {...props} />;
    }
    return null;
  };
  return (
    <Container>
      <CamVideo Id={animalId} />
      <p className="red">
        <img src="/red.png" />
      </p>
      <p className="yellow">
        <img src="/yellow.png" />
      </p>
      <div>
        {score === -1 ? (
          <AsyncImage
            className="cam"
            src={"http://localhost:5003/video_feed/" + animalId + "/" + level}
            alt="Video"
          />
        ) : (
          <div>
            <p className="font1">
              <h1>{score}</h1>
            </p>
            <p className="font2">
              <h2>{wrongList}</h2>
            </p>
          </div>
        )}
      </div>
      <p>
        <Link className="goback" to="/select">
          <img src="/goback.png" />
        </Link>
      </p>
      <p>
        <Link className="onemore" onClick={refreshPage}>
          <img src="/onemore.png" />
        </Link>
      </p>
    </Container>
  );
}

export default Cam;
