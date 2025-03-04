import unittest

from model_server.model_endpoint.retryable_model import RetryableModel


class TestSanitizeJson(unittest.TestCase):
    def test_sanitize_json(self):
        parsed = RetryableModel.parse_as_json("""
        ```json
{
  "codeResult": {
    "data": "```java\nimport java.io.IOException;\nimport java.nio.file.Files;\nimport java.nio.file.Path;\nimport java.nio.file.Paths;\nimport java.util.ArrayList;\nimport java.util.List;\n\npublic class CommitDiffSearcher {\n\n    public static List<String> findCommitDiffs(String projectDirectory, String commitMessage) throws IOException {\n        List<String> commitDiffs = new ArrayList<>();\n        Path projectPath = Paths.get(projectDirectory);\n\n        // Find all files modified in the commit\n        // This part requires access to git history.  Replace with your actual git command execution\n        //  This is a placeholder; you'll need to adapt this to your specific git command.\n        ProcessBuilder pb = new ProcessBuilder(\"git\", \"diff-tree\", \"-r\", \"--name-only\", \"HEAD\");\n        pb.directory(projectPath.toFile());\n        Process process = pb.start();\n        String diffFiles = new String(process.getInputStream().readAllBytes());\n        String[] files = diffFiles.split(\"\\n\");\n\n        // Iterate through the files and add their content to the commitDiffs list\n        for (String file : files) {\n            Path filePath = projectPath.resolve(file);\n            if (Files.exists(filePath)) {\n                String fileContent = Files.readString(filePath);\n                commitDiffs.add(fileContent);\n            }\n        }\n\n        return commitDiffs;\n    }\n\n    public static void main(String[] args) throws IOException {\n        String projectDirectory = \"test_graph_next\"; // Replace with your project directory\n        String commitMessage = \"message!\"; // Replace with the actual commit message\n        List<String> diffs = findCommitDiffs(projectDirectory, commitMessage);\n        System.out.println(diffs);\n    }\n}\n```"
  }
}
```
        """)


        assert parsed
        parsed = RetryableModel.parse_as_json("""
        ```json
{
  "codeResult": {
    "data": "```java\nimport java.io.IOException;\nimport java.nio.file.Files;\nimport java.nio.file.Path;\nimport java.nio.file.Paths;\nimport java.util.List;\nimport java.util.stream.Collectors;\n\npublic class GitDiffSearcher {\n\n    public static void main(String[] args) throws IOException {\n        // Replace with your project directory\n        Path projectDir = Paths.get(\"test_graph_next\");\n\n        // Get all files in the project directory\n        List<Path> files = Files.walk(projectDir)\n                .filter(Files::isRegularFile)\n                .collect(Collectors.toList());\n\n        // Filter files based on commit message (replace with your logic)\n        List<Path> relevantFiles = files.stream()\n                .filter(file -> file.getFileName().toString().contains(\"commit-diff-context\"))\n                .collect(Collectors.toList());\n\n        // Process relevant files (replace with your logic to extract diffs)\n        for (Path file : relevantFiles) {\n            String content = Files.readString(file);\n            // Extract diffs from content\n            System.out.println(\"File: \" + file.toString());\n            System.out.println(\"Content: \" + content);\n        }\n    }\n}\n```"
  }
}
```
""")
        assert parsed

        to_parse_code = """
        ```json
{
  "type": "code - add code identifier to specify to use code deserializer",
  "codeResult": {
    "data": [
      {
        "type": "file_change",
        "toChange": "file:///Users/hayde/IdeaProjects/drools/src/test/java/com/hayden/test_graph/commit_diff_context/step_def/LlmValidationNextCommit.java",
        "newContent": {
          "type": "insert_content",
          "linesToAdd": {
            "start": 287,
            "end": 287
          },
          "lines": [
            "import com.hayden.test_graph.commit_diff_context.model.server.ModelServerValidationAiClient;"
          ]
        }
      }
    ]
  }
}
```
        """

