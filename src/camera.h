#include <Core/vk_types.h>
#include <SDL_events.h>

#include <glm/gtx/quaternion.hpp>


class Camera
{
public:
    Camera();
    Camera(const glm::vec3& position_in, float pitch, float yaw);
    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();

    void processSDLEvent(SDL_Event& e);

    void update(float dt);

    void setSpeed(float speed_in);
    void setMouseSenstivity(float mouseSensitivity_in);

    bool isDirty() const;
    void clearDirtyBit();
public:
    glm::quat orientation;
    glm::vec3 position;
    glm::vec3 velocity; // relative to camera space
    bool rightMouseButtonDown = false; // to rotate the camera when right-click is pressed
    float speed;
    float mouseSensitivity;
private:
    struct MovementState
    {
        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    };
    MovementState movement;

    bool dirtyBit = false;
};