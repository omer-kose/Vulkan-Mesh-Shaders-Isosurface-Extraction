#include "camera.h"
#include <glm/gtx/transform.hpp>

Camera::Camera()
    :
    position(glm::vec3(0.0f, 0.0f, 0.0f)),
    orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
    speed(0.05f),
    mouseSensitivity(0.05f),
    velocity(glm::vec3(0.0f, 0.0f, 0.0f))

{}

Camera::Camera(const glm::vec3& position_in, float pitch, float yaw)
    :
    position(glm::vec3(0.0f, 0.0f, 0.0f)),
    speed(0.05f),
    mouseSensitivity(0.05f),
    velocity(glm::vec3(0.0f, 0.0f, 0.0f))
{
    glm::quat yawRot = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat pitchRot = glm::angleAxis(glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    orientation = orientation * yawRot;
    orientation = orientation * pitchRot;
    orientation = glm::normalize(orientation); // Prevent drift
    position += glm::vec3(glm::toMat4(orientation) * glm::vec4(position_in, 0.0f));
}

glm::mat4 Camera::getViewMatrix()
{
    glm::mat4 invTranslation = glm::translate(glm::mat4(1.0f), -position);
    glm::mat4 invRot = glm::transpose(glm::toMat4(orientation));
    return invRot * invTranslation;
}

glm::mat4 Camera::getRotationMatrix()
{
    return glm::toMat4(orientation);
}

void Camera::processSDLEvent(SDL_Event& e)
{
    if(e.type == SDL_KEYDOWN)
    {
        if(e.key.keysym.sym == SDLK_w) { movement.forward = true; }
        if(e.key.keysym.sym == SDLK_s) { movement.backward = true; }
        if(e.key.keysym.sym == SDLK_a) { movement.left = true; }
        if(e.key.keysym.sym == SDLK_d) { movement.right = true; }
        if(e.key.keysym.sym == SDLK_SPACE) { movement.up = true; }
        if(e.key.keysym.sym == SDLK_LCTRL) { movement.down = true; }
    }

    if(e.type == SDL_KEYUP)
    {
        if(e.key.keysym.sym == SDLK_w) { movement.forward = false; }
        if(e.key.keysym.sym == SDLK_s) { movement.backward = false; }
        if(e.key.keysym.sym == SDLK_a) { movement.left = false; }
        if(e.key.keysym.sym == SDLK_d) { movement.right = false; }
        if(e.key.keysym.sym == SDLK_SPACE) { movement.up = false; }
        if(e.key.keysym.sym == SDLK_LCTRL) { movement.down = false; }
    }

    if(e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT)
    {
        rightMouseButtonDown = true;
    }
    else if(e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT)
    {
        rightMouseButtonDown = false;
    }
    else if(e.type == SDL_MOUSEMOTION && rightMouseButtonDown)
    {
        float yawDelta = -e.motion.xrel * mouseSensitivity;
        float pitchDelta = -e.motion.yrel * mouseSensitivity;

        // World-space Y axis (global up)
        glm::quat yawRot = glm::angleAxis(glm::radians(yawDelta), glm::vec3(0.0f, 1.0f, 0.0f));

        // Camera's local X axis (right)
        glm::vec3 right = glm::rotate(orientation, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::quat pitchRot = glm::angleAxis(glm::radians(pitchDelta), right);

        // Apply yaw first (global), then pitch (local)
        orientation = yawRot * orientation;
        orientation = pitchRot * orientation;

        orientation = glm::normalize(orientation);
    }
}

void Camera::update()
{
    // Calculate net velocity based on all active keys
    glm::vec3 netVelocity(0.0f);
    if(movement.forward) netVelocity.z -= speed;
    if(movement.backward) netVelocity.z += speed;
    if(movement.left) netVelocity.x -= speed;
    if(movement.right) netVelocity.x += speed;
    if(movement.up) netVelocity.y += speed;
    if(movement.down) netVelocity.y -= speed;

    float netSpeed = glm::length(netVelocity);

    // Normalize if diagonal movement to maintain consistent speed
    if(netSpeed > speed)
    {
        netVelocity = glm::normalize(netVelocity) * speed;
    }

    // possible override above branch but not really important
    if(netSpeed > 0.0001f) // if there is a movement
    {
        // Apply movement relative to camera orientation
        position += glm::vec3(glm::toMat4(orientation) * glm::vec4(netVelocity, 0.0f));
    }
}

void Camera::setSpeed(float speed_in)
{
    speed = speed_in;
}

void Camera::setMouseSenstivity(float mouseSensitivity_in)
{
    mouseSensitivity = mouseSensitivity_in;
}
