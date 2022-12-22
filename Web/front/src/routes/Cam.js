import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";

function Cam() {
  const { animalId, level } = useParams();
  const [loading, setLoading] = useState(null);
  const [score, setScore] = useState(-1);
  const [wrongList, setWrongList] = useState([]);
  const [finish, setFinish] = useState(false);

  const getResult= () => {
    fetch(`http://localhost:5002/video_feed/${animalId}/${level}/fin`)
      .then(res=>{
        if(res.statue != 200) {
        }
        return res.json();
      })
      .then(data => {
        setScore(data['score'])
        setWrongList(data['wa'])
      })
  };

  useEffect(()=>{
    const timer = setInterval(getResult,3000);
    return () => { clearInterval(timer) }
  },[])

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
    <div>
        <AsyncImage src={"http://localhost:5002/video_feed/"+animalId+"/"+level} alt="Video" />
        <div>
            {
              score!==-1 && (<div>{score}{wrongList}</div>)
            }
        
        </div>
    </div>
  );
}

export default Cam;