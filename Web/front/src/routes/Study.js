import styled from "styled-components";
import { useParams } from "react-router-dom";
import "./styles.css";
import Video from "../components/Video";

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
    </Container>
  );
}

export default Study;
