import styled from "styled-components";
import "./styles.css";
import Video from "../components/Video";
import { Link } from "react-router-dom";

function Select() {
  const Container = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("/background_2.png");
    background-size: cover;
  `;
  return (
    <Container>
      <div className="rudolph">
        <Link to="/study/1">
          <img src="/rudolph.png" />
        </Link>
      </div>
      <div className="bear">
        <Link to="/study/2">
          <img src="/bear.png" />
        </Link>
      </div>
      <div className="rabbit">
        <Link to="/study/3">
          <img src="/rabbit.png" />
        </Link>
      </div>
      <div className="butterfly">
        <Link to="/study/4">
          <img src="/butterfly.png" />
        </Link>
      </div>
    </Container>
  );
}

export default Select;
