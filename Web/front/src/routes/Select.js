import styled from "styled-components";
import "./styles.css";
import video1 from "../video/butterfly.mov";

function Select() {
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
      <div className="video">
        <video width="1000" height="600" controls="controls">
          <source src={video1} type="video/mp4" />
        </video>
      </div>
    </Container>
  );
}

export default Select;
