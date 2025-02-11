import React, { useState } from "react";
import { Layout, Select, Typography, Splitter, FloatButton, Spin, Button, message } from "antd";
import { RightSquareOutlined, SyncOutlined, CopyOutlined, CheckOutlined } from "@ant-design/icons";
import MonacoEditor from "@monaco-editor/react";
import { refactorCode } from "../services/api";
import CodeRefactorRequest from "../interfaces/CodeRefactorRequest";
import CodeRefactorResponse from "../interfaces/CodeRefactorResponse";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark, okaidia} from "react-syntax-highlighter/dist/esm/styles/prism";

const { Header, Content } = Layout;
const { Title } = Typography;

const CodeAetherUI: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("generic");
  const [language, setLanguage] = useState("javascript");
  const [code, setCode] = useState("// Paste your code here...");
  const [refactoredCode, setRefactoredCode] = useState("Refactored code appears here...");

  const requestData: CodeRefactorRequest = {
    originalCode: code,
    language: language,
    modelType: model,
  };

  const handleRefactor = async () => {
    setLoading(true);
    try {
      const response: CodeRefactorResponse = await refactorCode(requestData);
      setRefactoredCode(response.refactoredCode);
      setLoading(false);
    } catch (error) {
      console.log(error);
      setLoading(false);
    }
  };

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      message.success("Copied to clipboard!");
    } catch (error) {
      message.error("Failed to copy!");
    }
  };

  return (
    <Layout style={{ height: "100vh" }}>
      {/* Navbar */}
      <Header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#001529", padding: "0 20px" }}>
        <Title level={3} style={{ color: "white", margin: 0 }}>CodeAether <RightSquareOutlined /> </Title>

        <div style={{ display: "flex", gap: "15px" }}>
          <Select value={model} onChange={setModel} style={{ width: 150 }}>
            <Select.Option value="generic">Generic Model</Select.Option>
            <Select.Option value="fine-tuned">Fine-Tuned Model</Select.Option>
          </Select>

          <Select value={language} onChange={setLanguage} style={{ width: 150 }}>
            <Select.Option value="javascript">JavaScript</Select.Option>
            <Select.Option value="python">Python</Select.Option>
            <Select.Option value="java">Java</Select.Option>
          </Select>
        </div>
      </Header>

      {/* Splitter Layout */}
      <Content style={{ padding: "20px", background: "#f0f2f5" }}>
        <Splitter>
          {/* Left Panel - Code Input */}
          <Splitter.Panel defaultSize={60} style={{ position: "relative" }}>
            <MonacoEditor
              height="calc(100vh - 64px)"
              language={language}
              theme="vs-dark"
              value={code}
              onChange={(value: any) => setCode(value || "")}
            />
            <FloatButton
              shape="square"
              type="primary"
              description={loading ? "Aethering..." : "Aether"}
              icon={loading ? <Spin indicator={<SyncOutlined style={{ fontSize: 20, color: "white" }} spin />} /> : <RightSquareOutlined />}
              style={{
                position: "absolute",
                bottom: "20px",
                right: "20px",
                width: "100px",
                height: "50px"
              }}
              onClick={handleRefactor}
            />
          </Splitter.Panel>

          {/* Right Panel - Refactored Code Output */}
          <Splitter.Panel defaultSize={40} style={{ padding: "15px", background: "#fff", overflowY: "auto", position: "relative" }}>
            <Title level={4}>Refactored Code</Title>

            {/* Markdown Renderer with Custom Code Blocks */}
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const codeText = String(children).replace(/\n$/, ""); // Extract pure text
                  return !inline ? (
                    <div style={{ position: "relative", marginBottom: "10px" }}>
                      <Button
                        type="text"
                        icon={<CopyOutlined />}
                        style={{ position: "absolute", top: 5, right: 5,color:'white' }}
                        onClick={() => handleCopy(codeText)}
                      />
                      <SyntaxHighlighter language={language} style={okaidia} {...props}>
                        {codeText}
                      </SyntaxHighlighter>
                    </div>
                  ) : (
                    <code {...props} className={className} style={{ background: "#f5f5f5", padding: "2px 5px", borderRadius: "3px" }}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {`\n${refactoredCode}\n\`\`\``}
            </ReactMarkdown>
          </Splitter.Panel>
        </Splitter>
      </Content>
    </Layout>
  );
};

export default CodeAetherUI;
