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




