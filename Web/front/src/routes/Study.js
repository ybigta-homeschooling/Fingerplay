import styled from "styled-components";
import Video from "../components/Video";
import { useParams, Link } from "react-router-dom";
import "./styles.css";

function Study() {
  const { animalId } = useParams();
  const imageData = {
    1: ["/videoimage/rudolph1_2.png", "/videoimage/rudolph1_1.png"],
    2: ["/videoimage/bear1.png", "/videoimage/bear1.png"],
    3: ["/videoimage/rabbit1_2.png", "/videoimage/rabbit1_1.png"],
    4: ["/videoimage/butterfly1_2.png", "/videoimage/butterfly1_1.png"],
  };
  const Container = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("/background_3.png");
    background-size: cover;
  `;
  return (
    <Container>
      <Video Id={animalId} />
      <p className="back">
        <img src="/back.png" />
      </p>
      <div className="do">
        <img src="/do_yourself.png" />
      </div>
      <div>
        <Link className="easy" to={"/cam/"+animalId+"/easy"}>
          <img src="/easy.png" />
        </Link>
      </div>
      <div>
        <Link className="hard" to={"/cam/"+animalId+"/hard"}>
          <img src="/hard.png" />
        </Link>
      </div>
      <p className="videoImage1">
        <img src={imageData[animalId][0]} />
      </p>
      <p className="videoImage2">
        <img src={imageData[animalId][1]} />
      </p>
    </Container>
  );
}

export default Study;
