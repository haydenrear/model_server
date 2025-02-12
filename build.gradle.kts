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

val enableDocker = project.property("enable-docker")?.toString()?.toBoolean()?.or(false) ?: false
val buildModelServer = project.property("build-model-server")?.toString()?.toBoolean()?.or(false) ?: false

println("enableDocker: $enableDocker, buildModelServer: $buildModelServer")

if (enableDocker && buildModelServer) {

    afterEvaluate {

        tasks.getByPath("jar").finalizedBy("buildDocker")

        tasks.getByPath("jar").doLast {
            tasks.getByPath("modelServerDockerImage").dependsOn("copyLibs")
            tasks.getByPath("pushImages").dependsOn("copyLibs")
        }

        tasks.register("buildDocker") {
            dependsOn("copyLibs", "bootJar", "modelServerDockerImage", "pushImages")
        }

        tasks.register("copyLibs") {
            println("Copying libs.")
            exec {
                workingDir(projectDir.resolve("docker"))
                commandLine("./build.sh")
            }
        }

        tasks.getByPath("pushImages").doLast {
            println("Pushing model server docker image.")
            exec {
                workingDir(projectDir.resolve("docker"))
                commandLine("./after-build.sh")
            }
        }

    }
}

dependencies {
    project(":runner_code")
}
