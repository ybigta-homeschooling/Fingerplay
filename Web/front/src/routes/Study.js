import styled from "styled-components";
import { useParams } from "react-router-dom";
import "./styles.css";
import Video from "../components/Video";
import Cam from "./Cam.js";
import { Link } from "react-router-dom";

function Study() {
  const { animalId } = useParams();
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
      <div className="do">
        <img src="/do_yourself.png" />
      </div>
      <div>
        <Link className="easy" to="./easy">
          <img src="/easy.png" />
        </Link>
      </div>
      <div>
        <Link className="hard" to="/cam">
          <img src="/hard.png" />
        </Link>
      </div>
    </Container>
  );
}

export default Study;
