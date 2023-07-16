import React, { useEffect, useState, useRef } from "react";
import { Container, Row, Col, Spinner } from "react-bootstrap";
import RangeSlider from "react-bootstrap-range-slider";
import { InferenceSession, Tensor } from "onnxruntime-web";
import fc3Onnx from "./assets/models/fc3.onnx";
import decoderOnnx from "./assets/models/decoder.onnx";

const createImageFromBytes = (canvas, bytes, height, width) => {
  const ctx = canvas.getContext("2d");
  canvas.height = height;
  canvas.width = width;

  const imageData = new ImageData(new Uint8ClampedArray(bytes), width, height);
  ctx.putImageData(imageData, 0, 0);

  return canvas;
};

const Image = ({ bytes }) => {
  const canvasRef = useRef();

  useEffect(() => {
    if (canvasRef.current) {
      createImageFromBytes(canvasRef.current, bytes, 256, 256);
    }
  }, [bytes]);

  return <canvas ref={canvasRef}></canvas>;
};

function App() {
  const [fc3Session, setFc3Session] = useState(null);
  const [decoderSession, setDecoderSession] = useState(null);
  const [imagesInTime, setImagesInTime] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [zIntTime, setZInTime] = useState([]);
  const [barValues, setBarValues] = useState([63, 14]);

  useEffect(() => {
    setZInTime(getSequenceParams(barValues[0], barValues[1]));
  }, []);

  useEffect(() => {
    async function loadModels() {
      try {
        const fc3Session = await InferenceSession.create(fc3Onnx);
        const decoderSession = await InferenceSession.create(decoderOnnx);

        setFc3Session(fc3Session);
        setDecoderSession(decoderSession);
      } catch (error) {
        console.log(error);
      }
    }

    loadModels();
  }, []);

  useEffect(() => {
    function generateImageData(z1, z2) {
      return new Promise((resolve, reject) => {
        if (zIntTime.length === 0 || !fc3Session || !decoderSession) {
          reject();
        }

        const z_data = Float32Array.from([z1, z2]);
        const z_tensor = new Tensor("float32", z_data, [2]);

        const fc3_feeds = { fc3_input0: z_tensor };
        fc3Session
          .run(fc3_feeds)
          .then((fc3_results) => {
            const fc3OutputData = fc3_results.fc3_output0.data;

            const decoder_feeds = {
              decoder_input0: new Tensor("float32", fc3OutputData, [1, 512]),
            };
            decoderSession
              .run(decoder_feeds)
              .then((decoder_results) => {
                const decoderOutputData = decoder_results.decoder_output0.data;

                const imageData = new Uint8ClampedArray(256 * 256 * 4);
                for (let i = 0; i < 256 * 256; ++i) {
                  const r = Math.round(255 * decoderOutputData[i]);
                  const g = Math.round(255 * decoderOutputData[256 * 256 + i]);
                  const b = Math.round(
                    255 * decoderOutputData[2 * 256 * 256 + i]
                  );
                  imageData[4 * i] = r;
                  imageData[4 * i + 1] = g;
                  imageData[4 * i + 2] = b;
                  imageData[4 * i + 3] = 255;
                }

                resolve(imageData);
              })
              .catch((error) => reject(error));
          })
          .catch((error) => reject(error));
      });
    }

    const generateImages = async () => {
      const imagesPromises = zIntTime.map(([z1, z2]) =>
        generateImageData(z1, z2)
      );

      try {
        const imagesData = await Promise.all(imagesPromises);
        setImagesInTime(imagesData);
      } catch (error) {
        console.log(error);
      }
    };

    if (zIntTime.length > 0 && fc3Session && decoderSession) {
      generateImages();
    }
  }, [zIntTime, fc3Session, decoderSession]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % imagesInTime.length);
    }, 1000 / 20);

    if (imagesInTime.length === 0) {
      clearInterval(interval);
    }
  }, [imagesInTime]);

  function harmonicOscillator(mu_z1, mu_z2, d, z1_circ, z2_circ, t) {
    var z1 = mu_z1 + d * Math.sin(t) * (z1_circ - mu_z1);
    var z2 = mu_z2 + d * Math.sin(t) * (z2_circ - mu_z2);
    return [z1, z2];
  }

  function createSequence(mu_z1, mu_z2, d, z1_circ, z2_circ, frames_cnt) {
    var z_buffer = [];
    for (var i = 0; i < frames_cnt; i++) {
      var t = (2 * Math.PI * i) / frames_cnt;
      var _harmonic_oscillator = harmonicOscillator(
        mu_z1,
        mu_z2,
        d,
        z1_circ,
        z2_circ,
        t
      );
      var z1 = _harmonic_oscillator[0];
      var z2 = _harmonic_oscillator[1];
      z_buffer.push([z1, z2]);
    }
    return z_buffer;
  }

  function getSequenceParams(param1, param2) {
    var t = param1 / 100;

    var z1_circ = -0.04814003;
    var z2_circ = 0.09167371;
    var R_circ = 2.21756703;
    var alpha = (R_circ * param2) / 100;
    var mu_z1 = z1_circ + R_circ * Math.sin(2 * Math.PI * (t + 0.5));
    var mu_z2 = z2_circ + R_circ * Math.cos(2 * Math.PI * (t + 0.5));
    const zIntTime_ = createSequence(
      mu_z1,
      mu_z2,
      alpha,
      z1_circ,
      z2_circ,
      100
    );
    return zIntTime_;
  }

  const handleBarValueChange = (index, value) => {
    const newBarValues = [...barValues];
    newBarValues[index] = value;
    setBarValues(newBarValues);

    const zIntTime_ = getSequenceParams(newBarValues[0], newBarValues[1]);
    setZInTime(zIntTime_);
    setImagesInTime([]);
  };

  const getExplanationText = (index) => {
    if (index === 0) {
      return "Time";
    } else if (index === 1) {
      return "Amplitude";
    }
  };

  const getLabelName = (index) => {
    if (index === 0) {
      return "Time";
    } else if (index === 1) {
      return "Amplitude";
    }
  };

  return (
    <Container>
      <Row
        style={{ marginTop: "20px", marginBottom: "20px", textAlign: "center" }}
      >
        <Col>
          <div style={{ margin: "20px", fontSize: "24px", fontWeight: "bold" }}>
            Variational Auto Encoder in React using onnx
          </div>
        </Col>
      </Row>
      <Row style={{ flexGrow: 1 }}>
        <Col md={4}>
          <div
            style={{
              textAlign: "right",
              width: "256px",
              height: "256px",
              border: "1px solid black",
              display: "flex",
            }}
          >
            {imagesInTime.length === 100 ? (
              <Image bytes={imagesInTime[currentIndex]} />
            ) : (
              <div
                style={{
                  textAlign: "center",
                  width: "256px",
                  height: "256px",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <Spinner animation="border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </Spinner>
              </div>
            )}
          </div>
        </Col>
        <Col md={4}>
          <div style={{ textAlign: "left" }}>
            {barValues.map((value, index) => (
              <div key={index}>
                <label
                  htmlFor={`input-range-${index}`}
                  data-toggle="tooltip"
                  data-placement="top"
                  title={getExplanationText(index)}
                  style={{ cursor: "pointer" }}
                >
                  {getLabelName(index)}
                </label>
                <RangeSlider
                  id={`input-range-${index}`}
                  min={1}
                  max={100}
                  value={value}
                  onChange={(e) =>
                    handleBarValueChange(index, parseInt(e.target.value))
                  }
                  tooltip="auto"
                />
              </div>
            ))}
          </div>
        </Col>
        <Col md={4}>
          <div style={{ margin: "20px", fontSize: "24px", fontWeight: "bold" }}>
            About
          </div>
          <div style={{ margin: "20px" }}>
            This is a simple example of a Variational Auto Encoder (VAE) in
            ReactJS using onnxruntime-web. All the frames of the candle are
            generated on the client side using the VAE. If you are interested in
            how the model was trained or how the site was built, please check
            out the
            <a href="https://github.com/detrin/VAE-candle-ONNX-react">
              GitHub repo
            </a>
            .
          </div>
        </Col>
      </Row>
    </Container>
  );
}

export default App;
