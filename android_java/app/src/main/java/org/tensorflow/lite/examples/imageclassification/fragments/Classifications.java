package org.tensorflow.lite.examples.imageclassification.fragments;

import org.tensorflow.lite.support.label.Category;

import java.util.List;

public class Classifications {
    private List<Category> categories;

    public Classifications(List<Category> categories) {
        this.categories = categories;
    }

    public List<Category> getCategories() {
        return categories;
    }
}
