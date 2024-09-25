import Com_hayden_docker_gradle.DockerContext;

plugins {
    id("com.hayden.docker")
}

wrapDocker {
    ctx = arrayOf(
        DockerContext(
            "localhost:5001/model-server",
            "${project.projectDir}/docker",
            "modelServer"
        )
    )
}

afterEvaluate {

    tasks.getByPath("modelServerDockerImage").dependsOn("copyLibs")
    tasks.getByPath("pushImages").dependsOn("copyLibs")

    tasks.register("copyLibs") {
        println("Copying libs.")
        exec {
            workingDir("docker")
            commandLine("./build.sh")
        }
    }

    tasks.getByPath("pushImages").doLast {
        exec {
            workingDir("docker")
            commandLine("./after-build.sh")
        }
    }
}


dependencies {
    project(":runner_code")
}
