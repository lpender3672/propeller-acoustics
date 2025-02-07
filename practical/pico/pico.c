#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"

#include "hardware/i2c.h"
#include "hardware/gpio.h"

#include "hx711-pico-c/include/common.h"



int main()
{
    stdio_init_all();

    // Initialise the Wi-Fi chip
    if (cyw43_arch_init()) {
        printf("Wi-Fi init failed\n");
        return -1;
    }

    // Example to turn on the Pico W LED
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    hx711_multi_config_t hxmcfg;
    hx711_multi_get_default_config(&hxmcfg);
    hxmcfg.clock_pin = 9;
    hxmcfg.data_pin_base = 12;
    hxmcfg.chips_len = 2;

    hx711_multi_t hxm;

    // 1. initialise
    hx711_multi_init(&hxm, &hxmcfg);

    // 2. Power up the HX711 chips and set gain on each chip
    hx711_multi_power_up(&hxm, hx711_gain_128);

    hx711_wait_settle(hx711_rate_80);


    int32_t arr[hxmcfg.chips_len];

    uint64_t time_in_microseconds;
    double time_in_seconds;

    while (true) {
        hx711_wait_settle(hx711_rate_80);
        hx711_multi_get_values(&hxm, arr);
        time_in_microseconds = time_us_64();
        time_in_seconds = time_in_microseconds / 1000000.0;
        
        printf("%f, %i, %i\n", time_in_seconds, arr[0], arr[1]);
    }
}
