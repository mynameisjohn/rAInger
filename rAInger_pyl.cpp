/*
This code was lifted from the tx_test sample in libloragw
I'm using it to send data over the MTAC antenna from python
*/


/* -------------------------------------------------------------------------- */
/* --- DEPENDANCIES --------------------------------------------------------- */

/* fix an issue between POSIX and C99 */
#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif

#include <stdint.h>		/* C99 types */
#include <stdbool.h>	/* bool type */
#include <stdio.h>		/* printf fprintf sprintf fopen fputs */

#include <string.h>		/* memset */
#include <signal.h>		/* sigaction */
#include <stdlib.h>		/* exit codes */

#include "loragw_hal.h"
#include "loragw_aux.h"

#define		RF_CHAIN		0	/* we'll use radio A only */
#define		DEFAULT_RSSI_OFFSET 	0.0

#include <algorithm>
#include <vector>
#include <iostream>
#include <pyliaison.h>

// Declare the packet template as global
// It gets configured in the init_radio function
struct lgw_pkt_tx_s g_TxPktTemplate;

// Init radio function, called from main
// most of the this code (basically all of it) was
// lifted from the libloragw tx_test example
bool init_radio();

// This is the function we export to pyliaision
// It sends us a bytearray type, but we interpret it as a char vec
bool send_loro_data( std::vector<char> vData )
{
	// copy template packet, but store our data
	struct lgw_pkt_tx_s txPkt = g_TxPktTemplate;
	txPkt.size = std::min<uint16_t>( vData.size(), sizeof( txPkt.payload ) );
	memcpy( txPkt.payload, vData.data(), txPkt.size );

	// send - this is a non-blocking call so don't bunch up...
	return LGW_HAL_SUCCESS == lgw_send( txPkt );
}

int main( int argc, char ** argv )
{
	// Try to init radio
	if ( !init_radio() )
	{
		std::cerr << "Error: Unable to initialize LoRaWAN radio" << std::endl;
		return EXIT_FAILURE;
	}

	// We may get an exception from the interpreter if something is amiss
	try
	{
		// Create python module with send_loro_data
		pyl::ModuleDef * pLLGWDef = pylCreateMod( pylLoRaWAN );
		if ( !pLLGWDef )
		{
			std::cerr << "Unable to create pyliaison module" << std::endl;
			return EXIT_FAILURE;
		}

		// Add our function to the module
		pylAddFnToMod( pLLGWDef, send_loro_data );

		// Initialize the python interpreter
		pyl::initialize();

		// Import the script, which will start the tflear
		// code as if we were running from main - we assume
		// the script is next to the executable...
		pyl::run_file( "rAInger.py" );

		// Shut down the interpreter
		pyl::finalize();

		/* clean up before leaving */
		lgw_stop();

		return EXIT_SUCCESS;
	}
	// These exceptions are thrown when something in pyliaison
	// goes wrong, but they're a child of std::runtime_error
	catch ( pyl::runtime_error e )
	{
		std::cerr << e.what() << std::endl;
		pyl::print_error();
		pyl::finalize();

		/* clean up before leaving */
		lgw_stop();

		return EXIT_FAILURE;
	}
}

// Function to initialize LoRaWAN MTAC radio
// Returns true if success, false if fail
// see util_tx_test.c in the lora_gateway repository
bool init_radio()
{
	/* board config */
	struct lgw_conf_board_s boardconf;
	memset( &boardconf, 0, sizeof( boardconf ) );
	boardconf.lorawan_public = true;
	lgw_board_setconf( boardconf );

	/* RF config */
	struct lgw_conf_rxrf_s rfconf;
	memset( &rfconf, 0, sizeof( rfconf ) );
	rfconf.enable = true;
	rfconf.freq_hz = 915;
	rfconf.rssi_offset = DEFAULT_RSSI_OFFSET;
	rfconf.type = LGW_RADIO_TYPE_SX1257;
	rfconf.tx_enable = true;
	lgw_rxrf_setconf( RF_CHAIN, rfconf );

	/* starting the concentrator */
	if ( lgw_start() != LGW_HAL_SUCCESS )
	{
		std::cerr << "ERROR: failed to start the concentrator" << std::endl;
		return false;
	}

	std::cout << "INFO: concentrator started, packets can be sent" << std::endl;

	/* fill-up payload and parameters */
	// These are all taken from othe tx_test example in libloragw
	// We'll use this packet as a template for all packets we send
	memset( &g_TxPktTemplate, 0, sizeof( g_TxPktTemplate ) );
	g_TxPktTemplate.freq_hz = 915;
	g_TxPktTemplate.tx_mode = IMMEDIATE;
	g_TxPktTemplate.rf_chain = RF_CHAIN;
	g_TxPktTemplate.rf_power = 14;
	g_TxPktTemplate.modulation = MOD_LORA;
	g_TxPktTemplate.bandwidth = BW_125KHZ;
	g_TxPktTemplate.datarate = DR_LORA_SF10;
	g_TxPktTemplate.coderate = CR_LORA_4_5;
	g_TxPktTemplate.invert_pol = 0;
	g_TxPktTemplate.preamble = 8;

	return true;
}
