import type { MenuItem, Restaurant } from "@/app/types/restaurant";

export type RestaurantHealthBreakdown = {
  proteinDensity: number;
  fiberDensity: number;
  fatQuality: number;
  calorieControl: number;
};

export type RestaurantHealthResult = {
  score: number;
  averageMenuQuality: number;
  breakdown: RestaurantHealthBreakdown;
};

export function calculateMenuItemHealthQuality(item: MenuItem): number {
  const breakdown = calculateMenuItemHealthBreakdown(item);

  return (
    breakdown.proteinDensity +
    breakdown.fiberDensity +
    breakdown.fatQuality +
    breakdown.calorieControl
  );
}

export function calculateMenuItemHealthBreakdown(
  item: MenuItem
): RestaurantHealthBreakdown {
  const proteinDensity = calculateProteinDensity(item);
  const fiberDensity = calculateFiberDensity(item);
  const fatQuality = calculateFatQuality(item);
  const calorieControl = calculateCalorieControl(item);

  return {
    proteinDensity,
    fiberDensity,
    fatQuality,
    calorieControl,
  };
}

export function calculateRestaurantHealthScore(
  restaurant: Restaurant
): RestaurantHealthResult {
  if (restaurant.menu.length === 0) {
    return {
      score: 0,
      averageMenuQuality: 0,
      breakdown: {
        proteinDensity: 0,
        fiberDensity: 0,
        fatQuality: 0,
        calorieControl: 0,
      },
    };
  }

  const menuBreakdowns = restaurant.menu.map((item) =>
    calculateMenuItemHealthBreakdown(item)
  );

  const averageProteinDensity = average(
    menuBreakdowns.map((item) => item.proteinDensity)
  );

  const averageFiberDensity = average(
    menuBreakdowns.map((item) => item.fiberDensity)
  );

  const averageFatQuality = average(
    menuBreakdowns.map((item) => item.fatQuality)
  );

  const averageCalorieControl = average(
    menuBreakdowns.map((item) => item.calorieControl)
  );

  const averageMenuQuality =
    averageProteinDensity +
    averageFiberDensity +
    averageFatQuality +
    averageCalorieControl;

  return {
    score: Math.round((averageMenuQuality / 40) * 100),
    averageMenuQuality: Math.round(averageMenuQuality),
    breakdown: {
      proteinDensity: Math.round(averageProteinDensity),
      fiberDensity: Math.round(averageFiberDensity),
      fatQuality: Math.round(averageFatQuality),
      calorieControl: Math.round(averageCalorieControl),
    },
  };
}

function calculateProteinDensity(item: MenuItem): number {
  const proteinPer100Calories =
    item.calories > 0 ? (item.protein / item.calories) * 100 : 0;

  const score = (proteinPer100Calories / 8) * 15;

  return Math.round(Math.max(0, Math.min(score, 15)));
}

function calculateFiberDensity(item: MenuItem): number {
  const fiberPer100Calories =
    item.calories > 0 ? (item.fiber / item.calories) * 100 : 0;

  const score = (fiberPer100Calories / 2) * 10;

  return Math.round(Math.max(0, Math.min(score, 10)));
}

function calculateFatQuality(item: MenuItem): number {
  const fatCalories = item.fat * 9;
  const fatCaloriesRatio =
    item.calories > 0 ? fatCalories / item.calories : 0;

  if (fatCaloriesRatio <= 0.25) {
    return 5;
  }

  if (fatCaloriesRatio <= 0.35) {
    return 3;
  }

  if (fatCaloriesRatio <= 0.45) {
    return 1;
  }

  return 0;
}

function calculateCalorieControl(item: MenuItem): number {
  if (item.calories <= 500) {
    return 10;
  }

  if (item.calories <= 650) {
    return 8;
  }

  if (item.calories <= 800) {
    return 5;
  }

  if (item.calories <= 950) {
    return 2;
  }

  return 0;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}