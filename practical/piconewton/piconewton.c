#include <stdio.h>
#include "pico/stdlib.h"

#include "hx711-pico-c/include/common.h"


int main()
{
    stdio_init_all();

    hx711_config_t thrust_amp_config;
    hx711_config_t torque_amp_config;
    hx711_get_default_config(&thrust_amp_config);
    hx711_get_default_config(&torque_amp_config);

    thrust_amp_config.clock_pin = 9;
    thrust_amp_config.data_pin = 12; 
    torque_amp_config.clock_pin = 9; 
    torque_amp_config.data_pin = 13;

    hx711_t thrust_amp;
    hx711_t torque_amp;
    hx711_init(&thrust_amp, &thrust_amp_config);
    hx711_init(&torque_amp, &torque_amp_config);

    // 3. Power up the hx711 and set gain on chip
    hx711_power_up(&thrust_amp, hx711_gain_128);
    hx711_power_up(&torque_amp, hx711_gain_128);

    int32_t thrust, torque;

    while (true) {
        // 6a. wait (block) until values are ready
        thrust = hx711_get_value(&thrust_amp);
        torque = hx711_get_value(&torque_amp);
        printf("thrust: %i torque: %i \n", thrust, torque);

    }
}
