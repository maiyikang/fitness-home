import { NextResponse } from "next/server";

import { adaptGooglePlacesToRestaurants } from "@/app/lib/restaurantAdapter";

export async function GET() {
  try {
    const apiKey = process.env.GOOGLE_PLACES_API_KEY;

    if (!apiKey) {
      return NextResponse.json(
        {
          error: "Missing GOOGLE_PLACES_API_KEY",
        },
        {
          status: 500,
        }
      );
    }

    const response = await fetch(
      "https://places.googleapis.com/v1/places:searchText",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Goog-Api-Key": apiKey,
          "X-Goog-FieldMask":
            "places.id,places.displayName,places.formattedAddress,places.primaryType",
        },
        body: JSON.stringify({
          textQuery: "restaurants in Southampton",
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();

      return NextResponse.json(
        {
          error: "Google Places request failed",
          details: errorText,
        },
        {
          status: response.status,
        }
      );
    }

    const data = await response.json();

    const restaurants =
      adaptGooglePlacesToRestaurants(
        data.places ?? []
      );

    return NextResponse.json(restaurants);
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to fetch places",
      },
      {
        status: 500,
      }
    );
  }
}