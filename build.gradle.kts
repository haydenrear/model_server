plugins {
    id("com.hayden.docker")
}

hayden_docker {
    imageName = "localhost:5001/model-server"
    contextDir = "${project.projectDir}/docker"
    enable = "true"
}

tasks.getByPath("dockerImage").dependsOn("copyLibs")
tasks.getByPath("pushImage").dependsOn("copyLibs")

tasks.register("copyLibs") {
    println("Copying libs.")
    exec {
        workingDir("docker")
        commandLine("./build-docker.sh")
    }
}

dependencies {
}
