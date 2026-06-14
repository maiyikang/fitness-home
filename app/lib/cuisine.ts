export function normalizeCuisine(value: string): string {
  return value.trim().toLowerCase();
}

export function getCuisineAliases(cuisine: string): string[] {
  const normalizedCuisine = normalizeCuisine(cuisine);

  const aliasMap: Record<string, string[]> = {
    healthy: ["healthy", "salad", "bowl", "fitness", "low fat"],
    sushi: ["sushi", "japanese", "roll", "sashimi", "nigiri"],
    burger: ["burger", "beef burger", "chicken burger", "fast food"],
    indian: [
      "indian",
      "curry",
      "tandoori",
      "biryani",
      "north indian",
      "south indian",
      "masala",
    ],
    chinese: [
      "chinese",
      "noodle",
      "dumpling",
      "rice bowl",
      "hotpot",
      "sichuan",
      "cantonese",
    ],
    korean: ["korean", "bibimbap", "kimchi", "bbq", "bulgogi"],
    italian: ["italian", "pizza", "pasta", "risotto"],
    mexican: ["mexican", "taco", "burrito", "quesadilla"],
  };

  return aliasMap[normalizedCuisine] ?? [normalizedCuisine];
}

export function isCuisineMatch(
  favoriteCuisines: string[],
  restaurantCuisine: string,
  restaurantCategory: string,
  restaurantTags: string[]
): boolean {
  const searchableText = [
    restaurantCuisine,
    restaurantCategory,
    ...restaurantTags,
  ]
    .join(" ")
    .toLowerCase();

  return favoriteCuisines.some((favoriteCuisine) => {
    const aliases = getCuisineAliases(favoriteCuisine);

    return aliases.some((alias) => {
      return searchableText.includes(alias);
    });
  });
}