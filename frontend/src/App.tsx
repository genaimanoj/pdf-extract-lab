import { Group, Panel, Separator } from "react-resizable-panels";
import { UploadBar } from "./components/UploadBar";
import { PdfPane } from "./components/PdfPane";
import { ResultPane } from "./components/ResultPane";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <UploadBar />
      <Group orientation="horizontal" className="main">
        <Panel defaultSize={50} minSize={25}>
          <div className="pane pane-pdf">
            <PdfPane />
          </div>
        </Panel>
        <Separator className="resize-handle" />
        <Panel defaultSize={50} minSize={25}>
          <div className="pane pane-result">
            <ResultPane />
          </div>
        </Panel>
      </Group>
    </div>
  );
}
