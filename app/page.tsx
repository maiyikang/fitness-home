"use client";

import { useState } from "react";

export default function Home() {
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [bodyFat, setBodyFat] = useState("");

  function handleStart() {
    console.log({
      height,
      weight,
      bodyFat,
    });
  }

  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10">
      <section className="mx-auto max-w-md rounded-2xl bg-white p-8 shadow-sm">
        <h1 className="text-4xl font-bold text-gray-900">
          Fitness Home
        </h1>

        <p className="mt-3 text-gray-600">
          Start with your body information.
        </p>

        <div className="mt-8 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Height
            </label>

            <input
              type="number"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              placeholder="Enter height in cm"
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Weight
            </label>

            <input
              type="number"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              placeholder="Enter weight in kg"
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Body Fat %
            </label>

            <input
              type="number"
              value={bodyFat}
              onChange={(e) => setBodyFat(e.target.value)}
              placeholder="Enter body fat percentage"
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <button
            onClick={handleStart}
            className="w-full rounded-lg bg-black px-6 py-3 font-medium text-white"
          >
            Start
          </button>
        </div>
      </section>
    </main>
  );
}
