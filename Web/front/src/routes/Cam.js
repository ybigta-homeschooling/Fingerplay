import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import styled from "styled-components";

function Cam() {
  const { animalId, level } = useParams();
  const [loading, setLoading] = useState(null);
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
    const timer = setInterval(getResult, 3000);
    return () => {
      clearInterval(timer);
    };
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
      {/* <p className="red">
        <img src="/red.png" />
      </p>
      <p className="yellow">
        <img src="/yellow.png" />
      </p> */}
      <div>
        <AsyncImage
          src={"http://localhost:5003/video_feed/" + animalId + "/" + level}
          alt="Video"
        />
        <div>
          {score !== -1 && (
            <div>
              {score}
              {wrongList}
            </div>
          )}
        </div>
      </div>
    </Container>
  );
}

export default Cam;
