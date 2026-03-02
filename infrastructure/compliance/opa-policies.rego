# TradeMaster OPA Policies
# Validates deployments for compliance

package trademaster.deployment

# Deny deployments without resource limits
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits
    msg := sprintf("Container '%s' must have resource limits", [container.name])
}

# Deny containers running as root
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%s' must not run as root", [container.name])
}

# Deny images without specific tags (no :latest)
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%s' must not use :latest tag", [container.name])
}

# Require health checks
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%s' must have a liveness probe", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.readinessProbe
    msg := sprintf("Container '%s' must have a readiness probe", [container.name])
}

# Deny exposed secrets in environment variables
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    env := container.env[_]
    contains(lower(env.name), "password")
    env.value
    msg := sprintf("Secret '%s' must use secretKeyRef, not plaintext value", [env.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    env := container.env[_]
    contains(lower(env.name), "secret")
    env.value
    msg := sprintf("Secret '%s' must use secretKeyRef, not plaintext value", [env.name])
}

# Require labels
deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.app
    msg := "Deployment must have 'app' label"
}

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.version
    msg := "Deployment must have 'version' label"
}

# Deny privileged containers
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.privileged
    msg := sprintf("Container '%s' must not be privileged", [container.name])
}
