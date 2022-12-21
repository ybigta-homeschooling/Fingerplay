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
      <span>
        <Link to="/study/1">
          <img src="/rudolph.png" className="rudolph" />
        </Link>
      </span>
      <span>
        <Link to="/study/2">
          <img src="/bear.png" className="bear" />
        </Link>
      </span>
      <span>
        <Link to="/study/3">
          <img src="/rabbit.png" className="rabbit" />
        </Link>
      </span>
      <span>
        <Link to="/study/4">
          <img src="/butterfly.png" className="butterfly" />
        </Link>
      </span>
    </Container>
  );
}

export default Select;
